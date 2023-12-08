import torch
import numpy as np
from scipy.io.wavfile import write
from preprocess import preprocess
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(description="Process model and input file paths.")
parser.add_argument(
    "--model_path",
    type=str,
    default="export/ddsp_cats_seg_pitchshift_pretrained.ts",
    help="Path to the model file.",
)
parser.add_argument(
    "--input_file",
    type=str,
    default="violin_original.mp3",
    help="Path to the input audio file.",
)
args = parser.parse_args()

# Load the model
model = torch.jit.load(args.model_path)

# Preprocess input file
x, pitch, loudness = preprocess(args.input_file, 16000, 160, 64000, False)
pitch = pitch[0]
loudness = loudness[0]
pitch = torch.tensor(pitch).unsqueeze(0).unsqueeze(-1).float()
loudness = torch.tensor(loudness).unsqueeze(0).unsqueeze(-1).float()

# Generate audio
audio = model(pitch, loudness)

# Save audio
audio_numpy = audio[0].detach().cpu().numpy()
# normalized_audio = audio_numpy / np.max(np.abs(audio_numpy))
audio_16bit = np.int16(audio_numpy * 32767)
write("test_output.wav", 16000, audio_16bit)
