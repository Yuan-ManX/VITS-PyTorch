import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import commons
import modules
import attentions
from commons import init_weights, get_padding


class StochasticDurationPredictor(nn.Module):
  """
  随机时长预测器（Stochastic Duration Predictor）。

  该类用于预测音频或语音合成中的时长信息，通过一系列的卷积流和仿射变换来处理输入特征，
  并输出对数时长信息。

  Args:
      in_channels (int): 输入特征的通道数。
      filter_channels (int): 滤波器的通道数。
      kernel_size (int): 卷积核的大小。
      p_dropout (float): Dropout概率，用于防止过拟合并提高模型的泛化能力。
      n_flows (int, optional): 卷积流的层数，默认为4。
      gin_channels (int, optional): 条件特征的通道数，如果为0，则不使用条件信息，默认为0。
  """
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()

    # 将 filter_channels 设置为 in_channels
    filter_channels = in_channels 

    # 初始化参数
    # 输入特征的通道数
    self.in_channels = in_channels
    # 滤波器的通道数
    self.filter_channels = filter_channels
    # 卷积核的大小
    self.kernel_size = kernel_size
    # Dropout概率
    self.p_dropout = p_dropout
    # 卷积流的层数
    self.n_flows = n_flows
    # 条件特征的通道数
    self.gin_channels = gin_channels

    # 定义对数函数层，用于计算对数流动
    self.log_flow = modules.Log()
    # 初始化卷积流模块列表
    self.flows = nn.ModuleList()
    # 添加元素级仿射变换层
    self.flows.append(modules.ElementwiseAffine(2))
    # 添加多个卷积流层和翻转层
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    # 定义输入预处理卷积层，将输入通道数从1扩展到 filter_channels
    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    # 定义后处理卷积层，保持通道数不变
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    # 定义深度可分离卷积层，用于进一步处理特征
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    # 初始化后处理卷积流模块列表
    self.post_flows = nn.ModuleList()
    # 添加元素级仿射变换层
    self.post_flows.append(modules.ElementwiseAffine(2))
    # 添加多个卷积流层和翻转层
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    # 定义预处理卷积层，将输入通道数从 in_channels 转换为 filter_channels
    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    # 定义投影卷积层，保持通道数不变
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    # 定义深度可分离卷积层，用于进一步处理特征
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    # 如果提供了条件特征，则定义一个条件卷积层
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    """
    前向传播方法，执行随机时长预测的计算。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, sequence_length)。
        x_mask (torch.Tensor): 输入的掩码张量，形状为 (batch_size, 1, sequence_length)。
        w (torch.Tensor, optional): 时长信息张量，形状为 (batch_size, 1, sequence_length)。
        g (torch.Tensor, optional): 条件信息张量，形状为 (batch_size, gin_channels, sequence_length)。
        reverse (bool, optional): 是否为反向传播，默认为False。
        noise_scale (float, optional): 噪声缩放因子，默认为1.0。

    Returns:
        torch.Tensor: 输出的对数时长信息或噪声张量。
    """
    # 将输入张量从计算图中分离，以防止梯度传播
    x = torch.detach(x)
    # 应用预处理卷积层
    x = self.pre(x)
    # 如果提供了条件信息，则将其添加到输入张量中
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    # 应用深度可分离卷积层
    x = self.convs(x, x_mask)
    # 应用投影卷积层，并应用掩码
    x = self.proj(x) * x_mask

    if not reverse:
      # 如果不是反向传播，则执行前向传播
      flows = self.flows
      # 确保时长信息张量存在
      assert w is not None

      # 初始化总对数流动
      logdet_tot_q = 0 
      # 应用后处理卷积层
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      # 生成随机噪声张量，并应用掩码
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      # 遍历后处理卷积流模块列表
      for flow in self.post_flows:
        # 应用卷积流层，并计算对数流动
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      # 将输出张量拆分为两个部分
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      # 应用sigmoid函数并应用掩码
      u = torch.sigmoid(z_u) * x_mask
      # 计算 z0
      z0 = (w - u) * x_mask
      # 计算对数流动
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      # 计算对数似然
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      # 重置总对数流动
      logdet_tot = 0
      # 应用对数流动层
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      # 合并张量
      z = torch.cat([z0, z1], 1)
      # 遍历前向卷积流模块列表
      for flow in flows:
        # 应用卷积流层，并计算对数流动
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      # 计算负对数似然
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      # 返回负对数似然和对数似然的和
      return nll + logq # [b]
    else:
      # 如果是反向传播，则反转卷积流列表顺序，并移除最后一个元素
      flows = list(reversed(self.flows))
      # 移除一个无用的vflow
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      # 生成随机噪声张量，并应用噪声缩放因子
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      # 遍历反转后的卷积流列表
      for flow in flows:
        # 应用卷积流层，并执行反向传播
        z = flow(z, x_mask, g=x, reverse=reverse)
      # 将输出张量拆分为两个部分
      z0, z1 = torch.split(z, [1, 1], 1)
      # 计算对数时长信息
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  """
    时长预测器（Duration Predictor）类。

    该类用于预测音频或语音合成中的时长信息，通过一系列的卷积层和归一化层处理输入特征，
    并输出对数时长信息。

    Args:
        in_channels (int): 输入特征的通道数。
        filter_channels (int): 滤波器的通道数。
        kernel_size (int): 卷积核的大小。
        p_dropout (float): Dropout概率，用于防止过拟合并提高模型的泛化能力。
        gin_channels (int, optional): 条件特征的通道数，如果为0，则不使用条件信息，默认为0。
    """
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    # 初始化参数
    # 输入特征的通道数
    self.in_channels = in_channels
    # 滤波器的通道数
    self.filter_channels = filter_channels
    # 卷积核的大小
    self.kernel_size = kernel_size
    # Dropout概率
    self.p_dropout = p_dropout
    # 条件特征的通道数
    self.gin_channels = gin_channels

    # 定义Dropout层
    self.drop = nn.Dropout(p_dropout)
    # 定义第一个卷积层，卷积核大小为 kernel_size，填充为 kernel_size//2 以保持尺寸
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    # 定义第一个层归一化层
    self.norm_1 = modules.LayerNorm(filter_channels)
    # 定义第二个卷积层，卷积核大小为 kernel_size，填充为 kernel_size//2 以保持尺寸
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    # 定义第二个层归一化层
    self.norm_2 = modules.LayerNorm(filter_channels)
    # 定义投影卷积层，将通道数从 filter_channels 转换为1
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    # 如果提供了条件特征，则定义一个条件卷积层
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    """
    前向传播方法，执行时长预测的计算。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, sequence_length)。
        x_mask (torch.Tensor): 输入的掩码张量，形状为 (batch_size, 1, sequence_length)。
        g (torch.Tensor, optional): 条件信息张量，形状为 (batch_size, gin_channels, sequence_length)。

    Returns:
        torch.Tensor: 输出的对数时长信息，形状为 (batch_size, 1, sequence_length)。
    """
    # 将输入张量从计算图中分离，以防止梯度传播
    x = torch.detach(x)
    if g is not None:
      # 将条件信息张量从计算图中分离
      g = torch.detach(g)
      # 将条件信息添加到输入张量中
      x = x + self.cond(g)

    # 应用第一个卷积层，并应用掩码
    x = self.conv_1(x * x_mask)
    # 应用ReLU激活函数
    x = torch.relu(x)
    # 应用第一个层归一化层
    x = self.norm_1(x)
    # 应用Dropout
    x = self.drop(x)
    # 应用第二个卷积层，并应用掩码
    x = self.conv_2(x * x_mask)
    # 应用ReLU激活函数
    x = torch.relu(x)
    # 应用第二个层归一化层
    x = self.norm_2(x)
    # 应用Dropout
    x = self.drop(x)
    # 应用投影卷积层，并应用掩码
    x = self.proj(x * x_mask)
    # 返回输出，并应用掩码
    return x * x_mask


class TextEncoder(nn.Module):
  """
  文本编码器（Text Encoder）类。

  该类用于将文本数据编码为高维表示，通过嵌入层、编码器层和投影层处理输入文本。

  Args:
      n_vocab (int): 词汇表的大小。
      out_channels (int): 输出特征的通道数。
      hidden_channels (int): 隐藏层的通道数。
      filter_channels (int): 滤波器的通道数。
      n_heads (int): 多头注意力的头数。
      n_layers (int): 编码器层的层数。
      kernel_size (int): 卷积核的大小。
      p_dropout (float): Dropout概率。
  """
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()

    # 初始化参数
    self.n_vocab = n_vocab  # 词汇表的大小
    self.out_channels = out_channels  # 输出特征的通道数
    self.hidden_channels = hidden_channels  # 隐藏层的通道数
    self.filter_channels = filter_channels  # 滤波器的通道数
    self.n_heads = n_heads  # 多头注意力的头数
    self.n_layers = n_layers  # 编码器层的层数
    self.kernel_size = kernel_size  # 卷积核的大小
    self.p_dropout = p_dropout  # Dropout概率

    # 定义嵌入层，将词汇索引转换为隐藏层表示
    self.emb = nn.Embedding(n_vocab, hidden_channels)
    # 使用正态分布初始化嵌入层权重
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    # 定义编码器，使用 `attentions` 模块中的 `Encoder` 类
    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    # 定义投影层，将隐藏层维度转换为输出通道数的两倍（用于均值和方差）
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    """
    前向传播方法，执行文本编码的计算。

    Args:
        x (torch.Tensor): 输入的文本张量，形状为 (batch_size, sequence_length)。
        x_lengths (torch.Tensor): 输入文本的长度，形状为 (batch_size,)。

    Returns:
        tuple: 包含以下内容的元组：
            - x (torch.Tensor): 编码后的文本特征，形状为 (batch_size, hidden_channels, sequence_length)。
            - m (torch.Tensor): 均值张量，形状为 (batch_size, out_channels, sequence_length)。
            - logs (torch.Tensor): 对数方差张量，形状为 (batch_size, out_channels, sequence_length)。
            - x_mask (torch.Tensor): 输入的掩码张量，形状为 (batch_size, 1, sequence_length)。
    """
    # 将词汇索引转换为嵌入向量，并乘以 sqrt隐藏层维度的平方根以进行缩放
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    # 转置张量形状为 (batch_size, hidden_channels, sequence_length)
    x = torch.transpose(x, 1, -1) # [b, h, t]
    # 生成掩码张量，形状为 (batch_size, 1, sequence_length)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    # 应用编码器，并应用掩码
    x = self.encoder(x * x_mask, x_mask)
    # 应用投影层，并应用掩码
    stats = self.proj(x) * x_mask

    # 将投影后的张量拆分为均值和方差
    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
  """
  残差耦合块（Residual Coupling Block）类。

  该类通过多个残差耦合层和翻转层（Flip）来构建一个残差耦合块，用于对输入张量进行变换。
  每个残差耦合层对输入进行部分变换，并通过残差连接来缓解梯度消失问题。

  Args:
      channels (int): 输入和输出的通道数。
      hidden_channels (int): 隐藏层的通道数。
      kernel_size (int): 卷积核的大小。
      dilation_rate (int): 膨胀率，用于控制卷积的感受野。
      n_layers (int): 残差耦合层中卷积层的层数。
      n_flows (int, optional): 残差耦合层的数量，默认为4。
      gin_channels (int, optional): 条件特征的通道数，如果为0，则不使用条件信息，默认为0。
  """
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()

    # 初始化参数
    self.channels = channels  # 输入和输出的通道数
    self.hidden_channels = hidden_channels  # 隐藏层的通道数
    self.kernel_size = kernel_size  # 卷积核的大小
    self.dilation_rate = dilation_rate  # 膨胀率
    self.n_layers = n_layers  # 残差耦合层中卷积层的层数
    self.n_flows = n_flows  # 残差耦合层的数量
    self.gin_channels = gin_channels  # 条件特征的通道数

    # 初始化模块列表，用于存储残差耦合层和翻转层
    self.flows = nn.ModuleList()

    # 添加多个残差耦合层和翻转层
    for i in range(n_flows):
      # 添加残差耦合层，参数包括通道数、隐藏通道数、卷积核大小、膨胀率、层数、条件通道数等
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      # 添加翻转层，用于交换输入张量的两个部分
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    """
    前向传播方法，执行残差耦合块的计算。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, channels, sequence_length)。
        x_mask (torch.Tensor): 输入的掩码张量，形状为 (batch_size, 1, sequence_length)。
        g (torch.Tensor, optional): 条件信息张量，形状为 (batch_size, gin_channels, sequence_length)。
        reverse (bool, optional): 是否为反向传播，默认为False。

    Returns:
        torch.Tensor: 经过残差耦合块处理后的输出张量，形状为 (batch_size, channels, sequence_length)。
    """
    if not reverse:
      for flow in self.flows:
        # 如果不是反向传播，则按顺序应用残差耦合层和翻转层
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      # 如果是反向传播，则逆序应用翻转层和残差耦合层
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class PosteriorEncoder(nn.Module):
  """
  后验编码器（Posterior Encoder）类。

  该类用于对输入张量进行编码，生成均值和方差参数，并通过随机采样生成潜在变量。
  它结合了卷积层、权重归一化层和投影层来处理输入特征。

  Args:
      in_channels (int): 输入特征的通道数。
      out_channels (int): 输出特征的通道数。
      hidden_channels (int): 隐藏层的通道数。
      kernel_size (int): 卷积核的大小。
      dilation_rate (int): 膨胀率，用于控制卷积的感受野。
      n_layers (int): 卷积层的层数。
      gin_channels (int, optional): 条件特征的通道数，如果为0，则不使用条件信息，默认为0。
  """
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()

    # 初始化参数
    self.in_channels = in_channels  # 输入特征的通道数
    self.out_channels = out_channels  # 输出特征的通道数
    self.hidden_channels = hidden_channels  # 隐藏层的通道数
    self.kernel_size = kernel_size  # 卷积核的大小
    self.dilation_rate = dilation_rate  # 膨胀率
    self.n_layers = n_layers  # 卷积层的层数
    self.gin_channels = gin_channels  # 条件特征的通道数

    # 定义预处理卷积层，将输入通道数从 in_channels 转换为 hidden_channels
    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    # 定义权重归一化层，用于对隐藏层特征进行归一化
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    # 定义投影层，将隐藏层维度转换为输出通道数的两倍（用于均值和方差）
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    """
    前向传播方法，执行后验编码器的计算。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, sequence_length)。
        x_lengths (torch.Tensor): 输入序列的长度，形状为 (batch_size,)。
        g (torch.Tensor, optional): 条件信息张量，形状为 (batch_size, gin_channels, sequence_length)。

    Returns:
        tuple: 包含以下内容的元组：
            - z (torch.Tensor): 潜在变量张量，形状为 (batch_size, out_channels, sequence_length)。
            - m (torch.Tensor): 均值张量，形状为 (batch_size, out_channels, sequence_length)。
            - logs (torch.Tensor): 对数方差张量，形状为 (batch_size, out_channels, sequence_length)。
            - x_mask (torch.Tensor): 输入的掩码张量，形状为 (batch_size, 1, sequence_length)。
    """
    # 生成掩码张量，形状为 (batch_size, 1, sequence_length)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    # 应用预处理卷积层，并应用掩码
    x = self.pre(x) * x_mask
    # 应用权重归一化层，并应用条件信息（如果提供）
    x = self.enc(x, x_mask, g=g)
    # 应用投影层，并应用掩码
    stats = self.proj(x) * x_mask
    # 将投影后的张量拆分为均值和方差
    m, logs = torch.split(stats, self.out_channels, dim=1)
    # 生成潜在变量，通过对均值添加随机噪声乘以方差
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    """
    生成器（Generator）类。

    该类实现了生成器网络，用于生成音频或语音信号。
    它由多个上采样层和残差块组成，并通过条件信息进行调节。

    Args:
        initial_channel (int): 初始输入通道数。
        resblock (str): 残差块的类型，'1' 或 '2'，对应不同的残差块实现。
        resblock_kernel_sizes (list of int): 残差块中卷积核的大小列表。
        resblock_dilation_sizes (list of int): 残差块中膨胀率的大小列表。
        upsample_rates (list of int): 上采样率列表。
        upsample_initial_channel (int): 上采样初始通道数。
        upsample_kernel_sizes (list of int): 上采样卷积核大小列表。
        gin_channels (int, optional): 条件特征的通道数，如果为0，则不使用条件信息，默认为0。
    """
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        # 残差块中卷积核的数量
        self.num_kernels = len(resblock_kernel_sizes)
        # 上采样层的数量
        self.num_upsamples = len(upsample_rates)
        # 定义预处理卷积层，卷积核大小为7，步幅为1，填充为3以保持尺寸
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        # 根据残差块的类型，选择相应的残差块实现
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        # 初始化上采样层列表
        self.ups = nn.ModuleList()
        # 遍历上采样率列表，添加上采样层
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        # 初始化残差块列表
        self.resblocks = nn.ModuleList()
        # 遍历上采样层列表，为每个上采样层添加多个残差块
        for i in range(len(self.ups)):
            # 计算当前上采样层的输出通道数
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                # 添加残差块
                self.resblocks.append(resblock(ch, k, d))

        # 定义后处理卷积层，卷积核大小为7，步幅为1，填充为3以保持尺寸，输出通道数为1
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # 应用权重归一化到所有上采样层
        self.ups.apply(init_weights)

        # 如果提供了条件通道数，则定义一个条件卷积层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        """
        前向传播方法，执行生成器的计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, initial_channel, sequence_length)。
            g (torch.Tensor, optional): 条件信息张量，形状为 (batch_size, gin_channels, sequence_length)。

        Returns:
            torch.Tensor: 生成的输出张量，形状为 (batch_size, 1, sequence_length)。
        """
        # 应用预处理卷积层
        x = self.conv_pre(x)
        if g is not None:
          # 如果提供了条件信息，则将其添加到输入张量中
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            # 应用LeakyReLU激活函数
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 应用上采样层
            x = self.ups[i](x)
            # 初始化残差块输出
            xs = None
            for j in range(self.num_kernels):
              if xs is None:
                # 应用残差块
                xs = self.resblocks[i*self.num_kernels+j](x)
              else:
                # 累加残差块输出
                xs += self.resblocks[i*self.num_kernels+j](x)
            # 对残差块输出取平均
            x = xs / self.num_kernels
        # 应用LeakyReLU激活函数
        x = F.leaky_relu(x)
        # 应用后处理卷积层
        x = self.conv_post(x)
        # 应用tanh激活函数
        x = torch.tanh(x)

        # 返回生成的输出张量
        return x

    def remove_weight_norm(self):
        """
        移除所有上采样层和残差块的权重归一化。

        该方法用于在训练完成后移除权重归一化，以减少推理时的计算开销。
        """
        print('Removing weight norm...')
        for l in self.ups:
            # 移除上采样层的权重归一化
            remove_weight_norm(l)
        for l in self.resblocks:
            # 移除残差块的权重归一化
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    """
    判别器P（DiscriminatorP）类。

    该类实现了判别器网络，用于区分真实音频样本和生成样本。
    它通过多个2D卷积层处理输入音频信号，并输出判别结果。
    该判别器将1D音频信号转换为2D表示，以便应用2D卷积操作。

    Args:
        period (int): 周期长度，用于将1D音频信号转换为2D表示。
        kernel_size (int, optional): 卷积核大小，默认为5。
        stride (int, optional): 卷积步幅，默认为3。
        use_spectral_norm (bool, optional): 是否使用谱归一化，默认为False。
    """
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        # 周期长度
        self.period = period
        # 是否使用谱归一化
        self.use_spectral_norm = use_spectral_norm
        # 选择归一化方法
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # 定义多个2D卷积层
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        # 定义后处理卷积层
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """
        前向传播方法，执行判别器的计算。

        Args:
            x (torch.Tensor): 输入的音频信号，形状为 (batch_size, channels, time_steps)。

        Returns:
            tuple:
                - torch.Tensor: 判别器的输出，形状为 (batch_size, 1)。
                - list of torch.Tensor: 每一层的特征映射列表。
        """
        # 初始化特征映射列表，用于存储每一层的输出
        fmap = []

        # 将1D音频信号转换为2D表示
        # 获取批次大小、通道数和序列长度
        b, c, t = x.shape
        # 如果序列长度不是周期的整数倍，则进行填充
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        # 重塑张量形状为 (batch_size, channels, time_steps / period, period)
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            # 应用卷积层
            x = l(x)
            # 应用LeakyReLU激活函数
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 将特征映射添加到列表中
            fmap.append(x)
        # 应用后处理卷积层
        x = self.conv_post(x)
        # 将后处理后的特征映射添加到列表中
        fmap.append(x)
        # 将张量展平为 (batch_size, channels * height * width)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    """
    判别器S（DiscriminatorS）类。

    该类实现了判别器网络，用于区分真实音频样本和生成样本。
    它通过多个1D卷积层处理输入音频信号，并输出判别结果。

    Args:
        use_spectral_norm (bool, optional): 是否使用谱归一化，默认为False。
    """
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        # 选择归一化方法
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # 定义多个1D卷积层
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        # 定义后处理卷积层
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        前向传播方法，执行判别器的计算。

        Args:
            x (torch.Tensor): 输入的音频信号，形状为 (batch_size, channels, time_steps)。

        Returns:
            tuple:
                - torch.Tensor: 判别器的输出，形状为 (batch_size, 1)。
                - list of torch.Tensor: 每一层的特征映射列表。
        """
        # 初始化特征映射列表，用于存储每一层的输出
        fmap = []

        for l in self.convs:
            # 应用卷积层
            x = l(x)
            # 应用LeakyReLU激活函数
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 将特征映射添加到列表中
            fmap.append(x)
        # 应用后处理卷积层
        x = self.conv_post(x)
        # 将后处理后的特征映射添加到列表中
        fmap.append(x)
        # 将张量展平为 (batch_size, channels * sequence_length)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    多周期判别器（MultiPeriodDiscriminator）类。

    该类实现了多周期判别器网络，通过多个判别器处理输入音频信号，并输出判别结果。
    它结合了判别器S和多个判别器P，以处理不同周期的音频信号。

    Args:
        use_spectral_norm (bool, optional): 是否使用谱归一化，默认为False。
    """
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()

        # 定义多个周期长度
        periods = [2,3,5,7,11]

        # 初始化判别器列表，首先添加判别器S
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        # 添加多个判别器P，每个判别器P使用不同的周期长度
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        # 将判别器列表转换为ModuleList
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        """
        前向传播方法，执行多周期判别器的计算。

        Args:
            y (torch.Tensor): 真实音频信号，形状为 (batch_size, channels, time_steps)。
            y_hat (torch.Tensor): 生成音频信号，形状为 (batch_size, channels, time_steps)。

        Returns:
            tuple:
                - list of torch.Tensor: 判别器对真实音频信号的判别结果列表。
                - list of torch.Tensor: 判别器对生成音频信号的判别结果列表。
                - list of list of torch.Tensor: 判别器对真实音频信号的特征映射列表。
                - list of list of torch.Tensor: 判别器对生成音频信号的特征映射列表。
        """
        y_d_rs = []  # 存储判别器对真实音频信号的判别结果
        y_d_gs = []  # 存储判别器对生成音频信号的判别结果
        fmap_rs = []  # 存储判别器对真实音频信号的特征映射
        fmap_gs = []  # 存储判别器对生成音频信号的特征映射

        # 遍历所有判别器
        for i, d in enumerate(self.discriminators):
            # 对真实音频信号进行判别
            y_d_r, fmap_r = d(y)
            # 对生成音频信号进行判别
            y_d_g, fmap_g = d(y_hat)
            # 将判别结果添加到列表中
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            # 将特征映射添加到列表中
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        # 返回判别结果和特征映射
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
  """
    训练用的合成器（Synthesizer for Training）。

    该类实现了用于训练的语音合成器，结合了文本编码器、生成器、后验编码器和流模块。
    它能够处理文本到语音的转换，并支持说话人嵌入以实现语音转换。

    Args:
        n_vocab (int): 词汇表的大小。
        spec_channels (int): 频谱特征的通道数。
        segment_size (int): 切片的段大小，用于训练时的随机切片。
        inter_channels (int): 中间层的通道数。
        hidden_channels (int): 隐藏层的通道数。
        filter_channels (int): 滤波器的通道数。
        n_heads (int): 多头注意力的头数。
        n_layers (int): 编码器或解码器的层数。
        kernel_size (int): 卷积核的大小。
        p_dropout (float): Dropout概率，用于防止过拟合并提高模型的泛化能力。
        resblock (str): 残差块的类型，'1' 或 '2'，对应不同的残差块实现。
        resblock_kernel_sizes (list of int): 残差块中卷积核的大小列表。
        resblock_dilation_sizes (list of int): 残差块中膨胀率的大小列表。
        upsample_rates (list of int): 上采样率列表。
        upsample_initial_channel (int): 上采样初始通道数。
        upsample_kernel_sizes (list of int): 上采样卷积核大小列表。
        n_speakers (int, optional): 说话人的数量，默认为0。
        gin_channels (int, optional): 条件特征的通道数，如果为0，则不使用条件信息，默认为0。
        use_sdp (bool, optional): 是否使用随机时长预测器，默认为True。
        **kwargs: 其他可选的关键字参数。
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers=0,
    gin_channels=0,
    use_sdp=True,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab  # 词汇表的大小
    self.spec_channels = spec_channels  # 频谱特征的通道数
    self.inter_channels = inter_channels  # 中间层的通道数
    self.hidden_channels = hidden_channels  # 隐藏层的通道数
    self.filter_channels = filter_channels  # 滤波器的通道数
    self.n_heads = n_heads  # 多头注意力的头数
    self.n_layers = n_layers  # 编码器或解码器的层数
    self.kernel_size = kernel_size  # 卷积核的大小
    self.p_dropout = p_dropout  # Dropout概率
    self.resblock = resblock  # 残差块的类型
    self.resblock_kernel_sizes = resblock_kernel_sizes  # 残差块中卷积核的大小列表
    self.resblock_dilation_sizes = resblock_dilation_sizes  # 残差块中膨胀率的大小列表
    self.upsample_rates = upsample_rates  # 上采样率列表
    self.upsample_initial_channel = upsample_initial_channel  # 上采样初始通道数
    self.upsample_kernel_sizes = upsample_kernel_sizes  # 上采样卷积核大小列表
    self.segment_size = segment_size  # 切片的段大小
    self.n_speakers = n_speakers  # 说话人的数量
    self.gin_channels = gin_channels  # 条件特征的通道数

    self.use_sdp = use_sdp  # 是否使用随机时长预测器

    # 初始化文本编码器
    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    
    # 初始化生成器
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    # 初始化后验编码器
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    # 初始化流模块（残差耦合块）
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

    # 如果使用随机时长预测器，则初始化随机时长预测器；否则，初始化时长预测器
    if use_sdp:
      self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
    else:
      self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

    # 如果说话人数量大于1，则初始化说话人嵌入层
    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

  def forward(self, x, x_lengths, y, y_lengths, sid=None):
    """
    前向传播方法，执行合成器的计算。

    Args:
        x (torch.Tensor): 输入文本张量，形状为 (batch_size, sequence_length)。
        x_lengths (torch.Tensor): 输入文本的长度，形状为 (batch_size,)。
        y (torch.Tensor): 输入的频谱特征，形状为 (batch_size, spec_channels, sequence_length)。
        y_lengths (torch.Tensor): 输入频谱特征的长度，形状为 (batch_size,)。
        sid (torch.Tensor, optional): 说话人ID，形状为 (batch_size,)。如果为None，则不使用说话人嵌入。

    Returns:
        tuple: 包含以下内容的元组：
            - o (torch.Tensor): 生成的频谱特征，形状为 (batch_size, upsample_initial_channel, sequence_length)。
            - l_length (torch.Tensor): 时长损失，形状为 (batch_size,)。
            - attn (torch.Tensor): 注意力权重，形状为 (batch_size, sequence_length, sequence_length)。
            - ids_slice (torch.Tensor): 切片ID，形状为 (batch_size,)。
            - x_mask (torch.Tensor): 输入文本的掩码，形状为 (batch_size, 1, sequence_length)。
            - y_mask (torch.Tensor): 输入频谱特征的掩码，形状为 (batch_size, 1, sequence_length)。
            - (z, z_p, m_p, logs_p, m_q, logs_q) (tuple): 包含潜在变量、均值和方差等信息的元组。
    """
    # 编码文本特征
    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    # 如果提供了说话人ID，则获取说话人嵌入；否则，设置为None
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    # 编码频谱特征
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
    # 应用流模块（残差耦合块）到频谱特征
    z_p = self.flow(z, y_mask, g=g)

    with torch.no_grad():
      # 计算负交叉熵
      s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

      # 生成注意力掩码
      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      # 计算单调对齐路径
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    # 计算时长信息
    w = attn.sum(2)
    if self.use_sdp:
      # 使用随机时长预测器计算时长损失
      l_length = self.dp(x, x_mask, w, g=g)
      l_length = l_length / torch.sum(x_mask)
    else:
      # 使用时长预测器计算时长损失
      logw_ = torch.log(w + 1e-6) * x_mask
      logw = self.dp(x, x_mask, g=g)
      l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask) # 用于归一化

    # 扩展先验分布
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    # 随机切片
    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    # 生成输出
    o = self.dec(z_slice, g=g)
    return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    """
    推理方法，执行语音合成的推理过程。

    Args:
        x (torch.Tensor): 输入文本张量，形状为 (batch_size, sequence_length)。
        x_lengths (torch.Tensor): 输入文本的长度，形状为 (batch_size,)。
        sid (torch.Tensor, optional): 说话人ID，形状为 (batch_size,)。如果为None，则不使用说话人嵌入。
        noise_scale (float, optional): 噪声缩放因子，默认为1。
        length_scale (float, optional): 时长缩放因子，默认为1。
        noise_scale_w (float, optional): 时长预测器的噪声缩放因子，默认为1。
        max_len (int, optional): 输出序列的最大长度。

    Returns:
        tuple: 包含以下内容的元组：
            - o (torch.Tensor): 生成的频谱特征，形状为 (batch_size, upsample_initial_channel, sequence_length)。
            - attn (torch.Tensor): 注意力权重，形状为 (batch_size, sequence_length, sequence_length)。
            - y_mask (torch.Tensor): 输入频谱特征的掩码，形状为 (batch_size, 1, sequence_length)。
            - (z, z_p, m_p, logs_p) (tuple): 包含潜在变量、均值和方差等信息的元组。
    """
    # 编码文本特征
    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    # 如果提供了说话人ID，则获取说话人嵌入；否则，设置为None
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    # 如果使用随机时长预测器，则使用随机时长预测器计算时长信息；否则，使用时长预测器
    if self.use_sdp:
      logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.dp(x, x_mask, g=g)

    # 计算时长信息
    w = torch.exp(logw) * x_mask * length_scale
    # 计算向上取整的时长信息
    w_ceil = torch.ceil(w)
    # 计算输出序列长度
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    # 生成输出掩码
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    # 生成注意力掩码
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    # 生成对齐路径
    attn = commons.generate_path(w_ceil, attn_mask)

    # 计算先验分布的均值和方差
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    # 对先验分布添加噪声
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    # 应用流模块（残差耦合块）到先验分布
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    # 生成输出
    o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    """
    语音转换方法，执行语音转换过程。

    Args:
        y (torch.Tensor): 输入的频谱特征，形状为 (batch_size, spec_channels, sequence_length)。
        y_lengths (torch.Tensor): 输入频谱特征的长度，形状为 (batch_size,)。
        sid_src (torch.Tensor): 源说话人ID，形状为 (batch_size,)。
        sid_tgt (torch.Tensor): 目标说话人ID，形状为 (batch_size,)。

    Returns:
        tuple: 包含以下内容的元组：
            - o_hat (torch.Tensor): 转换后的频谱特征，形状为 (batch_size, upsample_initial_channel, sequence_length)。
            - y_mask (torch.Tensor): 输入频谱特征的掩码，形状为 (batch_size, 1, sequence_length)。
            - (z, z_p, z_hat) (tuple): 包含潜在变量、潜在变量先验和转换后的潜在变量等信息的元组。
    """
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    # 获取源和目标说话人嵌入
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    # 编码输入频谱特征
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    # 应用流模块（残差耦合块）到输入频谱特征
    z_p = self.flow(z, y_mask, g=g_src)
    # 应用反向流模块（残差耦合块）到先验分布
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    # 生成转换后的输出
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

