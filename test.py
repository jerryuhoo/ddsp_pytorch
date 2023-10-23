import torch
from preprocess import preprocess
import librosa as li

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

write("test_output2.wav", 16000, audio[0].detach().cpu().numpy())
