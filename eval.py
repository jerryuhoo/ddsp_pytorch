import os
import torch
from preprocess import preprocess
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm
import torch.nn.functional as F

model = torch.jit.load("export/ddsp_mytraining_pretrained.ts")


def stft_mse_loss(
    original_audio,
    predicted_audio,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
    window="hamming",
):
    """
    Compute the MSE loss between the STFT of original and predicted audios.
    """
    orig_stft = torch.stft(
        original_audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(original_audio.device),
        return_complex=True,
    )
    pred_stft = torch.stft(
        predicted_audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(predicted_audio.device),
        return_complex=True,
    )

    # Taking magnitude of complex STFT values for MSE calculation
    orig_stft_magnitude = torch.abs(orig_stft)
    pred_stft_magnitude = torch.abs(pred_stft)

    return F.mse_loss(orig_stft_magnitude, pred_stft_magnitude)


def compute_reconstruction_loss(original_audio, predicted_audio):
    original_audio = original_audio.unsqueeze(1)
    loss = F.mse_loss(predicted_audio, original_audio)
    return loss.item()


directory = "data/cats/test"
output_directory = "data/cats/test_output"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

pitch_file_path = os.path.join(directory, "pitchs.npy")
loudness_file_path = os.path.join(directory, "loudness.npy")
signal_file_path = os.path.join(directory, "signals.npy")

# Check if pitch and loudness numpy files exist
if (
    os.path.exists(pitch_file_path)
    and os.path.exists(loudness_file_path)
    and os.path.exists(signal_file_path)
):
    print("Loading pitch and loudness from numpy files...")
    all_pitchs = np.load(pitch_file_path)
    all_loudness = np.load(loudness_file_path)
    all_x = np.load(signal_file_path)
else:
    print("Computing pitch and loudness...")
    all_pitchs = []
    all_loudness = []
    all_x = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            x, pitch, loudness = preprocess(filepath, 16000, 160, 64000, False)
            all_pitchs.append(pitch)
            all_loudness.append(loudness)
            all_x.append(x)
    # save pitchs and loudness as numpy arrays
    all_pitchs = np.concatenate(all_pitchs, 0).astype(np.float32)
    all_loudness = np.concatenate(all_loudness, 0).astype(np.float32)
    all_x = np.concatenate(all_x, 0).astype(np.float32)
    np.save(pitch_file_path, all_pitchs)
    np.save(loudness_file_path, all_loudness)
    np.save(signal_file_path, all_x)

total_loss = 0
total_stft_loss = 0
file_count = 0

for idx in tqdm(range(all_pitchs.shape[0])):
    pitch = torch.tensor(all_pitchs[idx]).float().unsqueeze(-1).unsqueeze(0)
    loudness = torch.tensor(all_loudness[idx]).float().unsqueeze(-1).unsqueeze(0)
    predicted_audio = model(pitch, loudness).squeeze()

    original_audio = torch.tensor(all_x[idx]).float().squeeze()

    # Compute the loss for the audio
    stft_loss = stft_mse_loss(original_audio, predicted_audio)
    loss = compute_reconstruction_loss(original_audio, predicted_audio)
    print(f"Loss for sample {idx}: {loss:.4f}")
    print(f"STFT Loss for sample {idx}: {stft_loss:.4f}")
    total_loss += loss
    total_stft_loss += stft_loss
    file_count += 1

    # Optionally save the predicted audio
    output_filepath = os.path.join(output_directory, f"predicted_{idx}.wav")
    write(output_filepath, 16000, predicted_audio.detach().cpu().numpy())


average_loss = total_loss / file_count
print(f"Average Reconstruction Loss: {average_loss:.4f}")

average_stft_loss = total_stft_loss / file_count
print(f"Average STFT Reconstruction Loss: {average_stft_loss:.4f}")

# save average loss to file
with open(os.path.join(output_directory, "average_loss.txt"), "w") as f:
    f.write(f"Average Reconstruction Loss: {average_loss:.4f}")
    f.write(f"Average STFT Reconstruction Loss: {average_stft_loss:.4f}")
