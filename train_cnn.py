import torch
from cnn_model_definition import ConvNet
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib

matplotlib.use('Agg')
from preprocessing import Data
import pickle


def top_k_accuracy(k, proba_pred_y, mini_y_test):
    top_k_pred = proba_pred_y.argsort(axis=1)[:, -k:]
    final_pred = [False] * len(mini_y_test)
    for j in range(len(mini_y_test)):
        final_pred[j] = True if sum(top_k_pred[j] == mini_y_test[j]) > 0 else False
    return np.mean(final_pred)


def import_and_concat_data(data_path, file_list):
    x, y = np.array([]), np.array([])
    for file_name in tqdm(file_list):
        temp_array = np.load(data_path + file_name)
        if file_name[0] == 'x':
            x = temp_array if x.size == 0 else np.concatenate([x, temp_array], axis=0)
        else:
            y = temp_array if y.size == 0 else np.concatenate([y, temp_array], axis=0)
    return x, y


filehandler = open('data/dataset.pth', 'rb')
dataset = pickle.load(filehandler)
dataset = Data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)

# setting model's parameters
learning_rate = model.get_learning_rate()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = torch.nn.CrossEntropyLoss()
epoch, batch_size = model.get_epochs(), model.get_batch_size()

n_total_steps = len(dataset.train_loader)
for epoch in range(model.get_epochs()):
    for i, (embedding, labels) in enumerate(dataset.train_loader):

        embedding = embedding.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(embedding)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{model.get_epochs()}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
