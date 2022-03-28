import torch
import torch.nn as nn
from preprocess.preprocessing import Data

DROP_OUT = 0.5
NUM_OF_CLASSES = 3


class ConvNet(nn.Module):

    def __init__(self, num_of_classes, dataset):
        super().__init__()
        # Hyper parameters
        self.epochs = 100
        self.batch_size = 28
        self.learning_rate = 0.001
        self.dataset = dataset
        # Model Architecture
        self.first_conv = nn.Conv2d(1, 96, kernel_size=(5, 5), padding=1)
        self.first_bn = nn.BatchNorm2d(96)
        self.first_polling = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.second_conv = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=1)
        self.second_bn = nn.BatchNorm2d(256)
        self.second_polling = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))

        self.third_conv = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.third_bn = nn.BatchNorm2d(384)

        self.forth_conv = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.forth_bn = nn.BatchNorm2d(256)

        self.fifth_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.fifth_bn = nn.BatchNorm2d(256)
        self.fifth_polling = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))

        self.sixth_conv = nn.Conv2d(256, 64, kernel_size=(2, 2), padding=1)
        self.first_drop = nn.Dropout(p=DROP_OUT)

        self.avg_polling = nn.AdaptiveAvgPool2d((1, 1))
        self.first_dense = nn.Linear(64, 1024)
        self.second_drop = nn.Dropout(p=DROP_OUT)

        self.second_dense = nn.Linear(1024, num_of_classes)

    def forward(self, X):
        x = nn.ReLU()(self.first_conv(X))
        x = self.first_bn(x)
        x = self.first_polling(x)

        x = nn.ReLU()(self.second_conv(x))
        x = self.second_bn(x)
        x = self.second_polling(x)

        x = nn.ReLU()(self.third_conv(x))
        x = self.third_bn(x)

        x = nn.ReLU()(self.forth_conv(x))
        x = self.forth_bn(x)

        x = nn.ReLU()(self.fifth_conv(x))
        x = self.fifth_bn(x)
        x = self.fifth_polling(x)

        x = nn.ReLU()(self.sixth_conv(x))
        x = self.first_drop(x)
        x = self.avg_polling(x)

        x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer

        x = nn.ReLU()(self.first_dense(x))
        x = self.second_drop(x)

        x = self.second_dense(x)
        y = nn.LogSoftmax(dim=1)(x)  # consider using Log-Softmax

        return y

    def get_epochs(self):
        return self.epochs

    def get_learning_rate(self):
        return self.learning_rate

    def get_batch_size(self):
        return self.batch_size

    def train_model(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([2.103336921, 3.187601958, 1]))

        n_total_steps = len(self.dataset.train_loader)

        for epoch in range(self.get_epochs()):
            for i, (embedding, labels) in enumerate(self.dataset.train_loader):

                embedding = embedding.type(torch.FloatTensor)
                labels = labels.type(torch.LongTensor)

                # Forward pass
                outputs = self.forward(embedding)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i == 86:
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')