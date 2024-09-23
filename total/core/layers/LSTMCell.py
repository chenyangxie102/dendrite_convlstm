import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    # __init__时类的构造函数，接受一些参数并且初始化LSTM单元的状态和参数
    # 包括输入通道数in_channel,隐藏单元数num_hidden,宽度width ，滤波器大小filter_size,步长stride，层规范层layer_norm.
    # 创建了两个卷积层conv_x和conv_y.并且进行了层规范处理
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(LSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )


    # LSTM单元的前向传播过程，它接受当前的输入x_t，以及上一时刻的隐藏状态h_t和单元状态c_t
    # 然后计算新的隐藏状态h_new和单元状态c_new.这个过程包括输入们、遗忘门、单元状态和输出们的计算，这些都是LSTM的标准组成部分。
    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)

        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h + c_new)
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new
