import os
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchaudio
import os
import sounddevice
from scipy.io.wavfile import write
from numpy import mat

from Model import ConvNet
from Dataset import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fine-tuning from wav2vec pytorch pipline
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

classes = {0:"Positive", 1:"Neutral" ,2:"Negative"}

def Norm(X):
    embedding = X.detach().cpu().numpy()
    for i in range(len(embedding)):
        mlist = embedding[0][i]
        embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
    return torch.from_numpy(embedding).to(device)


def recording(name):
	filename = name
	fps = 16000
	duration = 3
	print("Recording ..")
	recording = sounddevice.rec(int(duration * fps), samplerate = fps, channels = 2)
	sounddevice.wait()
	print("Done.")
	write(filename+".wav", fps, recording)
	return filename + ".wav"



def inference(file_name):
    waveform, sample_rate = torchaudio.load(recording(file_name))
    waveform = waveform.to(device)
    waveform = waveform.view(1,96000)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    with torch.inference_mode():
        embedding, _ = model(waveform)
        embedding = embedding.unsqueeze(0)
    return Norm(embedding)


if __name__ == '__main__':
    cnn = torch.load("dadaNet.pth" , map_location = torch.device("cpu"))
    cnn.eval()
    with torch.no_grad():
    	y = cnn(inference("example10"))
    y = y.cpu().detach().numpy()
    predict = [np.exp(c) for c in y]
    max = np.argmax(predict)
    sum = np.sum(predict)
    for_or = [np.round(100*c,3) for c in predict]
    print(for_or)
    print(classes[max])
   
