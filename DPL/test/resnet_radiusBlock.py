import numpy as np
import torch
import torch.nn as nn
'''-------------一、BasicBlock模块-----------------------------'''

# 用于ResNet18和ResNet34基本残差结构块
class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),  # inplace=True表示进行原地操作，一般默认为False，表示新建一个变量存储操作
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        # 论文中模型架构的虚线部分，需要下采样
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)  # 这是由于残差块需要保留原始输入
        out += self.shortcut(x)  # 这是ResNet的核心，在输出上叠加了输入x
        out = self.relu(out)
        return out
if __name__=='__main__':
    input = torch.randn(4, 64, 256, 234)  # 随机生成一个输入特征图
    test=BasicBlock(64,64)  # 实例化残差模块，设置输出输入通道为64
    output = test(input)  # 将输入特征图通过残差模块进行处理
    print(output.shape)  # 打印处理后的特征图形状，验证残差模块的作用



# # ResNet-18/34 残差结构 BasicBlock
# class BasicBlock(nn.Module):
#     expansion = 1   # 残差结构中主分支所采用的卷积核的个数是否发生变化。对于浅层网络，每个残差结构的第一层和第二层卷积核个数一样，故是1
#
#     # 定义初始函数
#     # in_channel输入特征矩阵深度，out_channel输出特征矩阵深度（即主分支卷积核个数）
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None):   # downsample对应虚线残差结构捷径中的1×1卷积
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, padding=1, bias=False)  # 使用bn层时不使用bias
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=1, padding=1, bias=False)  # 实/虚线残差结构主分支中第二层stride都为1
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.downsample = downsample   # 默认是None
#
# # 定义正向传播过程
#     def forward(self, x):
#         identity = x   # 捷径分支的输出值
#         if self.downsample is not None:   # 对应虚线残差结构
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)   # 这里不经过relu激活函数
#
#         out += identity
#         out = self.relu(out)
#
#         return out


