import torch
from cnn_model_definition import ConvNet
import numpy as np
import os
from tqdm import tqdm
import datetime
import matplotlib
from preprocessing import Data
import pickle
import datetime


def top_k_accuracy(k, proba_pred_y, mini_y_test):
    top_k_pred = proba_pred_y.argsort(axis=1)[:, -k:]
    final_pred = [False] * len(mini_y_test)
    for j in range(len(mini_y_test)):
        final_pred[j] = True if sum(top_k_pred[j] == mini_y_test[j]) > 0 else False
    return np.mean(final_pred)


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

for epoch in range(0):
    for i, (embedding, labels) in enumerate(dataset.train_loader):

        embedding = embedding.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        # Forward pass
        outputs = model(embedding)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 40 == 0:
            print(f'Epoch [{epoch + 1}/{model.get_epochs()}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

saved_results = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(7)]
    n_class_samples = [0 for i in range(7)]
    for embedding, labels in dataset.test_loader:
        embedding = embedding.to(device)

        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        outputs = model(embedding)

        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    saved_results.append(f'Accuracy of the network: {acc} %')

    for i in range(7):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {dataset.classes[i]}: {acc} %')
        saved_results.append(f'Accuracy of {dataset.classes[i]}: {acc} %')

saved_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
file_name = 'result.txt'
directory = 'data/' + str(saved_time)
os.mkdir(directory)

with open(directory + "/" + file_name, 'w') as f:
    for line in saved_results:
        f.write(line)
        f.write('\n')

torch.save(model, directory + "/model.pth")
