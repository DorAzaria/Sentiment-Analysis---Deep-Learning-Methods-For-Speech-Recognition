"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn
import constants as c
import torch

DROP_OUT = 0.5
DIMENSION = 512 * 300


class ConvNet(nn.Module):


    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5, 5), padding=1), nn.ReLU(), nn.BatchNorm2d(96), nn.MaxPool2d(3, 2),
            # first convolutional layer
            # [96,147,30] -> after max polling : [96,73,14]

            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(3, 1),
            # second convolutional layer
            # [256,71,12] -> after max polling : [256,69,10]

            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(384),
            # third convolutional layer
            # [384,69,10]

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            # forth convolutional layer
            # [256,69,10]

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2)),
            # fifth convolutional layer
            # [256,69,10] -> after max polling : [256,22,4]

            nn.Conv2d(256, 64, kernel_size=(2, 2), padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            # sixth convolutional layer
            # [64,23,5]
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            # first dense layer
            nn.Linear(64, 1024), nn.ReLU(), nn.Dropout(p=0.5),
            # second dense layer
            nn.Linear(1024, 7), nn.ReLU(), nn.LogSoftmax(dim=1),)

    def forward(self, X):
        return self.network(X)

    def get_epochs(self):
        return 200

    def get_learning_rate(self):
        return 0.0001

    def get_batch_size(self):
        return 16

    def to_string(self):
        return "Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_"
