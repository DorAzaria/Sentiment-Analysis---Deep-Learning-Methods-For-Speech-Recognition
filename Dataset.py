import numpy as np
import pandas as pd
import os
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class Data:

    def __init__(self):
        file_handler = open('/content/data/MyDrive/dl/dataset5.pth', 'rb')
        self.data = pickle.load(file_handler)
        x_dataset = [embedding[1] for embedding in self.data]
        y_dataset = [label[2] for label in self.data]

        #[70, 15, 15]
        train_x, rem_x, train_y, rem_y = train_test_split(np.array(x_dataset), np.array(y_dataset), train_size=0.70) 
        valid_x, test_x, valid_y, test_y = train_test_split(rem_x, rem_y, test_size=0.5)

        self.train_x = torch.from_numpy(train_x)
        self.train_y = torch.from_numpy(train_y)
        torch_train = TensorDataset(self.train_x, self.train_y)

        self.valid_x = torch.from_numpy(valid_x)
        self.valid_y = torch.from_numpy(valid_y)
        torch_valid = TensorDataset(self.valid_x, self.valid_y)
        
        self.test_x = torch.from_numpy(test_x)
        self.test_y = torch.from_numpy(test_y)
        torch_test = TensorDataset(self.test_x, self.test_y)
        
        self.train_loader = DataLoader(torch_train, batch_size=32, drop_last=True, shuffle=True)
        self.valid_loader = DataLoader(torch_valid, batch_size=32, drop_last=True, shuffle=True)
        self.test_loader = DataLoader(torch_test, batch_size=32, drop_last=True, shuffle=False)
