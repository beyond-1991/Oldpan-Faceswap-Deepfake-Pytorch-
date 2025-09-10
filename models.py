import torch
import torch.utils.data
from torch import nn, optim
from padding_same_conv import Conv2d




#正常的原始形状是(batch_size, height, width, channels)
# img.transpose((0, 3, 1, 2)) 的作用是channels放到第二个位置，height放到第三个位置，width放到第四个位置
# 这是因为pytorch的卷积层的输入是(batch_size, channels, height, width)的形状
# 而我们的输入是(batch_size, height, width, channels)的形状，所以需要进行转置

#之后torch.from_numpy()的作用是将numpy数组转换为torch张量
# 最后返回的是一个torch张量，形状是(batch_size, channels, height, width)
def toTensor(img):
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img

# img_var.data.cpu().numpy()的作用是将torch张量转换为numpy数组
# data.cpu()的作用是将torch张量转换为cpu张量
def var_to_np(img_var):
    return img_var.data.cpu().numpy()


class _ConvLayer(nn.Sequential):
    def __init__(self, input_features, output_features):
        super(_ConvLayer, self).__init__()
        self.add_module('conv2', Conv2d(input_features, output_features,
                                        kernel_size=5, stride=2))
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))
# LeakyReLU的作用是解决ReLU函数的梯度消失问题
# 当输入小于0时，LeakyReLU会返回一个很小的负数： k(斜率=0.1) * 输入，而不是0
# 这样可以避免梯度消失问题，同时保持网络的非线性



class _UpScale(nn.Sequential): # 通道减半、尺寸翻倍
    def __init__(self, input_features, output_features):
        super(_UpScale, self).__init__()
        self.add_module('conv2_', Conv2d(input_features, output_features * 4,
                                         kernel_size=3))
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))
        self.add_module('pixelshuffler', _PixelShuffler())
# 先把输入通道扩大4倍 目的准备足够的马赛克块 
# 第二 给这些信息加上点”非线性变化“ 让特征更丰富
# 执行的搭积木操作 最终输出放大两倍的图，通道恢复正常


class Flatten(nn.Module):

    def forward(self, input):
        output = input.reshape(input.size(0), -1)
        return output


class Reshape(nn.Module):

    def forward(self, input):
        output = input.view(-1, 1024, 4, 4)  # channel * 4 * 4

        return output


class _PixelShuffler(nn.Module):
    def forward(self, input):
        batch_size, c, h, w = input.size() # batch_size, c, h, w = input.size() # 输入的形状是(batch_size, c, h, w)

        rh, rw = (2, 2) # 放大倍数
        oh, ow = h * rh, w * rw # 输出高度和宽度

        oc = c // (rh * rw) # 输入通道数除以4
        out = input.view(batch_size, rh, rw, oc, h, w) # 输入的形状是(batch_size, rh, rw, oc, h, w)

        out = out.permute(0, 3, 4, 1, 5, 2).contiguous() # 重新排列为(batch_size, oc, h, w, rh, rw)

        out = out.view(batch_size, oc, oh, ow)  # 最后重塑为(batch_size, oc, oh, ow)


        return out


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            _ConvLayer(3, 128), 
            _ConvLayer(128, 256),   
            _ConvLayer(256, 512),  
            _ConvLayer(512, 1024),   
            # 并不致命
            Flatten(),
            nn.Linear(1024 * 4 * 4,1024), # 强行要求的  # 这里是有点问题，如果有知道请告诉我
            nn.Linear(1024,1024 * 4 * 4),
            Reshape(),
            _UpScale(1024, 512),
        )

        self.decoder_A = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

        self.decoder_B = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, select='A'):
        if select == 'A':
            out = self.encoder(x)
            out = self.decoder_A(out)
        else:
            out = self.encoder(x)
            out = self.decoder_B(out)
        return out
