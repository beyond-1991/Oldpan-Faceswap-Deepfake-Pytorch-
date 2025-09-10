# modify con2d function to use same padding
# code referd to @famssa in 'https://github.com/pytorch/pytorch/issues/3867'
# and tensorflow source code

import torch.utils.data
from torch.nn import functional as F

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# custom con2d, because pytorch don't have "padding='same'" option.
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)  # 输入的特征图高度
    filter_rows = weight.size(2)    # 输入的卷积核高度
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1 # 计算卷积核有效高度 考虑到空洞率 它会扩大卷积核的实际覆盖情况
    out_rows = (input_rows + stride[0] - 1) // stride[0]  # 计算输出的特征图高度
    # 动态计算padding 
    # 方式1：用有效滤波器尺寸计算
    # padding_needed = max(0,  (out_rows - 1) * stride[0] + effective_filter_size_rows -input_rows)
    # 方式2：用原始卷积核尺寸+空洞率计算（与方式1等价）# 计算padding高度 # （out_rows-1 :第一次不需要滑动）， * 步长
    padding_rows = max(0,  (out_rows - 1) * stride[0] +(filter_rows - 1) * dilation[0] + 1 - input_rows)

    # 计算是不是padding  奇数？
    rows_odd = (padding_rows % 2 != 0)
    # 【注意】这里代码复用了高度方向的变量（out_rows/filter_rows/input_rows），而非宽度变量（out_cols/filter_cols/input_cols）
    # 实际逻辑应为：用宽度参数计算（out_cols/filter_cols/input_cols），但代码中按高度参数计算，等价于“宽度padding=高度padding”
    # 示例：与padding_rows计算相同 → padding_cols=3
    padding_cols = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)

    # 【注意】这里用padding_rows判断宽度是否为奇数，而非padding_cols，等价于“宽度是否为奇数=高度是否为奇数”
    # 示例：padding_rows=3 → cols_odd=True
    cols_odd = (padding_rows % 2 != 0)

    # 补边规则：[左, 右, 上, 下] → 这里只补右和下（各补1行/列，当为奇数时）
    # 示例：rows_odd=True, cols_odd=True → 输入从(2,3,256,256)变为(2,3,257,257)（高度+1，宽度+1）   
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    # 改写完padding , 调用pytorch的conv2d
    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)
