"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn
import constants as c
import torch

DROP_OUT = 0.5
DIMENSION = 512 * 300


class ConvNet(nn.Module):

    def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
        return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2

    def __init__(self):

        super().__init__()

        self.conv_2d_1 = nn.Conv2d(1, 16, kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.bn_1 = nn.BatchNorm2d(16)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        # [16,74,16]

        self.conv_2d_2 = nn.Conv2d(16, 32, kernel_size = (3, 3), stride=(1, 1), padding=1)
        self.bn_2 = nn.BatchNorm2d(32)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        # [32,36,8]

        self.conv_2d_3 = nn.Conv2d(32, 64, kernel_size = (3, 3),  stride = (1, 1), padding = 1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.drop_3 = nn.Dropout(p = DROP_OUT)
        # [64,18,4]

        self.dense_1 = nn.Linear(4608, 1024)
        self.drop_2 = nn.Dropout(p = DROP_OUT)

        self.dense_2 = nn.Linear(1024, 8)

    def forward(self, X):

        x = nn.ReLU()(self.conv_2d_1(X))
        x = self.bn_1(x)
        x = self.max_pool_2d_1(x)

        x = nn.ReLU()(self.conv_2d_2(x))
        x = self.bn_2(x)
        x = self.max_pool_2d_2(x)

        x = nn.ReLU()(self.conv_2d_3(x))
        x = self.bn_3(x)
        x = self.max_pool_2d_3(x)

        x = x.view(28, -1)  # output channel for flatten before entering the dense layer
        x = nn.ReLU()(self.dense_1(x))

        x = self.dense_2(x)
        y = nn.LogSoftmax(dim = 1)(x)   # consider using Log-Softmax

        return y

    def get_epochs(self):
        return 700

    def get_learning_rate(self):
        return 0.0001

    def get_batch_size(self):
        return 16

    def to_string(self):
        return "Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_"
