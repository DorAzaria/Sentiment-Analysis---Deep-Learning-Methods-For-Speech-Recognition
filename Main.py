import os
import IPython
import numpy as np
import torch
import torchaudio
import sounddevice
from scipy.io.wavfile import write
from numpy import mat
from Model import ConvNet

SAMPLE_RATE = 16000

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# fine-tuning from wav2vec pytorch pipline
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

classes = {0: "Positive", 1: "Neutral", 2: "Negative"}


def Norm(X):
    embedding = X.detach().cpu().numpy()
    for i in range(len(embedding)):
        mlist = embedding[0][i]
        embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
    return torch.from_numpy(embedding).to(device)


def recording(name):
    filename = name
    duration = 3
    print("Recording ..")
    recording = sounddevice.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sounddevice.wait()
    print("Done.")
    write(filename + ".wav", SAMPLE_RATE, recording)
    return filename + ".wav"


def inference(file_name):
    waveform, sr = torchaudio.load(recording(file_name), num_frames = SAMPLE_RATE*3)
    
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    waveform = waveform.to(device)

    return waveform


def print_results(y):
    y = y.cpu().detach().numpy()
    predict = [np.exp(c) for c in y]
    max = np.argmax(predict)
    print(f'Predicted: {classes[max].capitalize()}')
    print(f'Positive: {round(predict[0][0] * 100, 4)}%')
    print(f'Neutral:  {round(predict[0][1] * 100, 4)}%')
    print(f'Negative: {round(predict[0][2] * 100, 4)}%')


if __name__ == '__main__':
    cnn = torch.load("dadaNet.pth", map_location=torch.device("cpu"))
    cnn.eval()

    with torch.inference_mode():
        tor = inference("test_recording")
        embedding, _ = model(tor)
        embedding = embedding.unsqueeze(0)
        embedding = Norm(embedding)
        y = cnn(embedding)
        print_results(y)
