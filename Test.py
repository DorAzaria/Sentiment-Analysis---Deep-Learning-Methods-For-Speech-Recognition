import Model
import torch
import datetime
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestConvNet:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.results = []

    def test(self):

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(3)]
            n_class_samples = [0 for i in range(3)]
            for embedding, labels in self.dataset.test_loader:

                embedding = labels.type(torch.FloatTensor)
                labels = labels.type(torch.LongTensor)

                outputs = self.model.forward(embedding)

                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(self.model.batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')
            self.results.append(f'Accuracy of the network: {acc} %')

            for i in range(3):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {self.dataset.classes[i]}: {acc} %')
                self.results.append(f'Accuracy of {self.dataset.classes[i]}: {acc} %')

        saved_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
        file_name = 'result.txt'
        directory = 'data/' + str(saved_time)
        os.mkdir(directory)

        with open(directory + "/" + file_name, 'w') as f:
            for line in self.results:
                f.write(line)
                f.write('\n')

        torch.save(self.model, directory + "/model.pth")
