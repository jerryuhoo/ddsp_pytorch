import os
import glob
import torch
from preprocess import preprocess
import librosa as li
from scipy.io.wavfile import write
from tqdm import tqdm

model = torch.jit.load("export/ddsp_mytraining_pretrained.ts")

folder_path = "/home/yifeng/SVS/data/violin_segments"

output_folder = "output_test"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files = []
for file_type in ["*.mp3", "*.wav"]:
    files.extend(glob.glob(os.path.join(folder_path, file_type)))

for file_path in tqdm(files, desc="Processing files"):
    x, pitch, loudness = preprocess(file_path, 16000, 160, 64000, False)
    pitch = pitch[0]
    loudness = loudness[0]
    pitch = torch.tensor(pitch).unsqueeze(0).unsqueeze(-1).float()
    loudness = torch.tensor(loudness).unsqueeze(0).unsqueeze(-1).float()

    audio = model(pitch, loudness)

    base_name = os.path.basename(file_path)
    output_file_name = os.path.splitext(base_name)[0] + ".wav"
    output_file_path = os.path.join(output_folder, output_file_name)

    write(output_file_path, 16000, audio[0].detach().cpu().numpy())
