from Model import ConvNet
from Test import TestConvNet
from preprocess.preprocessing import Data

if __name__ == '__main__':
    aer_dataset = Data()
    cnn = ConvNet(3, aer_dataset)
    cnn.train_model()
    test = TestConvNet(cnn, aer_dataset)
    test.test()
