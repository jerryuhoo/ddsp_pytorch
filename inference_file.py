import torch
from preprocess import preprocess
import librosa as li
import numpy as np

model = torch.jit.load("export/ddsp_mytraining_pretrained.ts")

f = "violin_original.mp3"

x, pitch, loudness = preprocess(f, 16000, 160, 64000, False)
print("pitch", pitch.shape)
print("loudness", loudness.shape)
pitch = pitch[0]
loudness = loudness[0]
pitch = torch.tensor(pitch).unsqueeze(0).unsqueeze(-1).float()
print("pitch", pitch.shape)
loudness = torch.tensor(loudness).unsqueeze(0).unsqueeze(-1).float()
print("loudness", loudness.shape)


audio = model(pitch, loudness)

# save audio
from scipy.io.wavfile import write

audio_numpy = audio[0].detach().cpu().numpy()
# normalized_audio = audio_numpy / np.max(np.abs(audio_numpy))
audio_16bit = np.int16(audio_numpy * 32767)
write("test_output1.wav", 16000, audio_16bit)
