import torch
from ConvNet import Convolutional_Speaker_Identification
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def top_k_accuracy(k, proba_pred_y, mini_y_test):
    top_k_pred = proba_pred_y.argsort(axis=1)[:, -k:]
    final_pred = [False] * len(mini_y_test)
    for j in range(len(mini_y_test)):
        final_pred[j] = True if sum(top_k_pred[j] == mini_y_test[j]) > 0 else False
    return np.mean(final_pred)


def import_and_concat_data(data_path , file_list):
    x, y = np.array([]), np.array([])
    for file_name in tqdm(file_list):
        temp_array = np.load(data_path + file_name)
        if file_name[0] == 'x':
            x = temp_array if x.size == 0 else np.concatenate([x, temp_array], axis=0)
        else:
            y = temp_array if y.size == 0 else np.concatenate([y, temp_array], axis=0)
    return x, y

def create_tensors(data_path , data_file_list , i,block_num):
    file_list = list(data_file_list[i*block_num: (i+1)*block_num].flatten())
    x, y = import_and_concat_data(data_path , file_list)
    y = pd.Series(y).str.split('id1', expand=True).iloc[:, -1].values.astype(int) - 1
    x = np.expand_dims(x, axis=1)
    return torch.from_numpy(x).float(), torch.from_numpy(y)

def random_file_data(path):
    data_file_list = os.listdir(path)
    file_mat = np.sort(data_file_list).reshape(2,int(len(data_file_list)/2)).T
    np.random.shuffle(file_mat)
    return file_mat


data_path = 'pickles\\'
training_results_path = 'results\\'

print('Start training:')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Convolutional_Speaker_Identification()

# setting model's parameters
learning_rate = get_learning_rate()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = torch.nn.CrossEntropyLoss()
epoch, batch_size = model.get_epochs(), get_batch_size()

# preparing txt report file
results_df = pd.DataFrame([], columns=['train_loss', 'val_loss', 'top_1_test_acc', 'top_5_test_acc', 'top_10_test_acc'])
now_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
dir_path = training_results_path + now_time
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

file = open(dir_path + '/reuslts.txt', 'w')
file_txt = ['Date and time :  ' + now_time,'Learning Rate : ' + str(learning_rate), 'Batch Size : ' + str(batch_size), 'Epoch Number : ' + str(epoch)]
for s in file_txt:
    file.write('\r----------------------------\r\r')
    file.write(s)

# Saving data
for e in range(epoch):

    model.train()
    train_loss, val_loss = 0, 0

    train_files = random_file_data(data_path + 'train_files_npy/')
    block_num = 3
    count_train = 0
    for g in range(int(np.ceil(len(train_files)/block_num))):

        x_train, y_train = create_tensors(data_path + 'train_files_npy/' , train_files , g,block_num)

        epoch_size = len(x_train)
        train_epoch_idx = np.random.choice(len(y_train), epoch_size, replace=False)
        np.random.shuffle(train_epoch_idx)

        batch_num = int(np.ceil(epoch_size / batch_size))

        for b in tqdm(range(batch_num)):
            optimizer.zero_grad()
            batch_loc = train_epoch_idx[(b * batch_size):((b + 1) * batch_size)]
            x_batch, y_batch = x_train[batch_loc], y_train[batch_loc]
            y_pred = model(x_batch.to(device))
            loss = criterion(y_pred, y_batch.long().to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            count_train+=1

    train_loss = np.round(train_loss/count_train, 4)

    # checking the model's performances per epoch

    with torch.no_grad():
        model.eval()

        val_files = random_file_data(data_path + 'val_files_npy/')
        count_val=0
        for g in range(int(np.ceil(len(val_files)/block_num))):
            x_val, y_val = create_tensors(data_path + 'val_files_npy/' , val_files , g, block_num)
            val_epoch_idx = np.random.choice(len(y_val), len(y_val), replace=False)
            for h in range(int(len(y_val) / batch_size) - 1):
                val_batch_loc = val_epoch_idx[(h * batch_size): ((h + 1) * batch_size)]
                mini_x_val, mini_y_val = x_val[val_batch_loc], y_val[val_batch_loc]
                y_pred_val = model(mini_x_val.to(device))
                val_loss += criterion(y_pred_val, mini_y_val.long().to(device)).item()
                count_val+=1
        val_loss = np.round(val_loss/count_val, 4)

        # calculating predictions on the test set
        final_accuracy = np.array([0, 0, 0], dtype=float)
        test_files = random_file_data(data_path + 'test_files_npy/')
        count_test = 0
        for g in range(int(len(test_files)/block_num) + 1):
            x_test, y_test = create_tensors(data_path + 'test_files_npy/' , test_files , g, block_num)
            test_epoch_idx = np.random.choice(len(y_test), len(y_test), replace=False)

            for l in range(int(np.ceil(len(y_test)/batch_size))):
                test_batch_loc = test_epoch_idx[(l * batch_size):((l + 1) * batch_size)]
                mini_x_test, mini_y_test = x_test[test_batch_loc], y_test[test_batch_loc]
                proba_pred_y = model(mini_x_test.to(device))
                count_test +=1
                accuracy_list = []

                for k in [1, 5, 10]:
                    accuracy_list += [top_k_accuracy(k, proba_pred_y, mini_y_test)]
                final_accuracy += np.array(accuracy_list)

        final_accuracy /= count_test

    #if 0 == (e % 5):
        #torch.save(model, dir_path + "/" + model.to_string() + str(e + 1) + ".pth")
    torch.save(model, dir_path + "/" + to_string() + str(e + 1) + ".pth")
    new_final_accuracy = [round(x * 100, 3) for x in list(final_accuracy)]

    results_df.loc[len(results_df)] = [train_loss, val_loss] + new_final_accuracy
    print('Epoch: ',e ,' -', dict(results_df.iloc[-1]))

    epoch_report = "Epoch : " + str(e + 1) + " | Train_Loss: " + str(train_loss) + ' , Val_Loss: ' + str(val_loss)
    file.write('\r----------------------------\r\r')
    file.write(epoch_report)

file.close()

results_df.to_excel(dir_path + '/final_report.xlsx')

plt.figure(figsize=(10, 10))
plt.plot(results_df['train_loss'], color='gold', label='train')
plt.plot(results_df['val_loss'], color='purple', label='val')
plt.ylabel('Loss', fontsize=25)
plt.xlabel('Epoch', fontsize=25)
plt.legend()
plt.savefig(dir_path + '/final_loss_plot.jpeg')
