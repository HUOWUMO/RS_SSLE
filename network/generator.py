import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import *
import cv2
from mtutils import min_max_normalize
import random

class Generator_Gaussian(nn.Module):

    def __init__(self, inchannel, patch_size, std=0.2, re_grad_weight=False, init_alpha=1, re_grad_alpha=False):
        super(Generator_Gaussian, self).__init__()
        self.inchannel = inchannel
        self.patch_size = patch_size
        # 设置2D高斯权重
        gaussian_kernel_1D = cv2.getGaussianKernel(patch_size, std, cv2.CV_32F)  # 构建一维高斯核,方差为std
        gaussian_kernel_2D = gaussian_kernel_1D * gaussian_kernel_1D.T  # 由一维高斯核构建二维高斯核
        gaussian_kernel_2D = min_max_normalize(gaussian_kernel_2D)  # 归一化,整体趋势不变，数值有效增加

        kernel = torch.FloatTensor(gaussian_kernel_2D).expand(inchannel, patch_size, patch_size)  # 复制扩展高斯核
        self.weight = nn.Parameter(data=kernel, requires_grad=re_grad_weight)  #

        self.alpha = nn.Parameter(torch.tensor(init_alpha), requires_grad=re_grad_alpha)  # [0.0,False]等价于原始鉴别器

    def forward(self, x):  # 前向传播默认test模式

        in_size = x.size(0)  # 注意epoch末尾的batch_size<=设置的固定值，因此要重新计算

        gaussian_weight = self.weight
        gaussian_weight_expand = gaussian_weight.expand(in_size, self.inchannel, self.patch_size,
                                                        self.patch_size)  # 复制扩展高斯核
        x_gaussian = x * gaussian_weight_expand

        x = (1 - self.alpha) * x + self.alpha * x_gaussian

        return x


class Synthesis(nn.Module):
    def __init__(self, N, ni, beta, num_iteration):
        super().__init__()

        self.N = N
        self.ni = ni
        self.beta = beta
        self.num_iteration = num_iteration

    def forward(self, y):

        # 第一轮迭代初始
        # 注意，如果不用.copy()，会直接把变量地址同化，和C++一样
        y_buff1 = y.copy()
        y_buff2 = y.copy()  # 后三个不参与迭代，直接赋予y的值

        y_buff2[0] = y_buff1[0]

        # 先移动特别的y[1]

        y_buff2[1] = y_buff1[1] - self.ni * (
                (1 - self.beta) * (-2 * y_buff1[0] + 5 * y_buff1[1] - 4 * y_buff1[2] + y_buff1[3]) + self.beta * (
                y_buff1[1] - y[1]))

        # 后面一视同仁
        for n in range(2, self.N - 2):
            y_buff2[n] = y_buff1[n] - self.ni * ((1 - self.beta) * (
                    1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 * y_buff1[
                n + 2]) + self.beta * (y_buff1[n] - y[n]))

        # 多轮迭代
        for k in range(0, self.num_iteration):

            y_buff1 = y_buff2.copy()
            y_buff2 = y.copy()

            y_buff2[0] = y_buff1[0]

            # 移动特别的y[1]
            y_buff2[1] = y_buff1[1] - self.ni * ((1 - self.beta) * (
                    -2 * y_buff1[0] + 5 * y_buff1[1] - 4 * y_buff1[2] + y_buff1[3]) + self.beta * (
                                                         y_buff1[1] - y[1]))

            # 后面一视同仁
            for n in range(2, self.N - 2):
                y_buff2[n] = y_buff1[n] - self.ni * ((1 - self.beta) * (
                        1 * y_buff1[n - 2] - 4 * y_buff1[n - 1] + 6 * y_buff1[n] - 4 * y_buff1[n + 1] + 1 *
                        y_buff1[n + 2]) + self.beta * (y_buff1[n] - y[n]))

        return y_buff2
