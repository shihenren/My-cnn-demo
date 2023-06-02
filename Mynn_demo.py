import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

'''
输入一个3通道图像，Conv1会使用32个不同的5*5大小卷积核进行卷积操作，
每个卷积核会生成一个特征图，最终输出一个由32个特征图组成的张量，
这些特征图是经过卷积运算提取原始图像中不同位置的特征得到的。
每个特征图的每个位置表示一个不同的局部特征，包括边缘、角点、纹理等信息。
padding设置为2，保持图像原尺寸不变
Conv3的output_channel为64
池化层核大小设置为2。共进行三次池化操作

先进行三次卷积和池化，然后进行展平，
'''
class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,input):
        output = self.model(input)
        return input

#简单验证网络结构
mynn = Mynn()
input = torch.ones((64,3,32,32))#测试数据
output = mynn(input)
print(output.shape)#输出结果和模型预期一致

#tensorboard展示模型
writer= SummaryWriter("logs")
writer.add_graph(mynn,input)
writer.close()