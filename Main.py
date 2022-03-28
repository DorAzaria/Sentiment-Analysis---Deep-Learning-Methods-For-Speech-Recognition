import os
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchaudio
import os

from numpy import mat

from Model import ConvNet
from Test import TestConvNet
from preprocess.preprocessing import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fine-tuning from wav2vec pytorch pipline
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)


def Norm(X):
    embedding = X.detach().cpu().numpy()
    for i in range(len(embedding)):
        mlist = embedding[0][i]
        embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
        if embedding[0][i] < -1 or embedding[0][i] > 1:
            print("NISHBAR HAZAIN")
            break
    return torch.from_numpy(embedding)


def recording(name):
    # import sounddevice
    # # from scipy.io.wavefile import write
    # filename = name
    # fps = 16000
    # duration = 3
    # print("Recording ..")
    # recording = sounddevice.rec(int(duration * fps), samplerate = fps, channels = 2)
    # sounddevice.wait()
    # print("Done.")
    # write(filename, fps, recording)
    # return filename + ".wav"
    pass


def inference(file_name):
    waveform, sample_rate = torchaudio.load(recording(file_name))
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    with torch.inference_mode():
        embedding, _ = model(waveform)

    return embedding


if __name__ == '__main__':
    aer_dataset = Data()
    cnn = ConvNet(3, aer_dataset)
    cnn.train_model()
    test = TestConvNet(cnn, aer_dataset)
    test.test()
    X = Norm(inference("dor_angry"))
    predict = [mat.exp(c) for c in cnn.forward(X)]
