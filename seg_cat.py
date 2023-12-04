import librosa
import librosa.display
import numpy as np
import os
import soundfile as sf
from scipy.signal import butter, filtfilt


def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    y = filtfilt(b, a, data)
    return y


def load_and_filter_audio(filename, sr=16000, cutoff=400):
    y, sr = librosa.load(filename, sr=sr)
    y_filtered = butter_highpass_filter(y, cutoff, sr)
    return y_filtered, sr


def calculate_energy(y, window_size, step_size):
    energy = []
    for i in range(0, len(y) - window_size, step_size):
        window_energy = np.sum(np.square(y[i : i + window_size]))
        energy.append((i, window_energy))
    return energy


def find_and_save_high_energy_segments(
    y, sr, energy, threshold, base_filename, window_size, output_folder
):
    current_segment_start = None
    current_segment_end = None
    segment_index = 0

    for i, e in energy:
        if e > threshold:
            if current_segment_start is None:
                current_segment_start = i
            current_segment_end = i + window_size
        else:
            if current_segment_start is not None:
                segment_audio = y[current_segment_start:current_segment_end]
                segment_filename = f"{base_filename}_seg_{segment_index}.wav"
                output_file_path = os.path.join(output_folder, segment_filename)
                sf.write(output_file_path, segment_audio, sr)
                segment_index += 1

                current_segment_start = None
                current_segment_end = None

    if current_segment_start is not None:
        segment_audio = y[current_segment_start:current_segment_end]
        segment_filename = f"{base_filename}_seg_{segment_index}.wav"
        output_file_path = os.path.join(output_folder, segment_filename)
        sf.write(output_file_path, segment_audio, sr)


folder_path = "/home/yifeng/SVS/ddsp_pytorch/data/cats/train"
output_folder = "/home/yifeng/SVS/ddsp_pytorch/data/cats/train_seg"
threshold = 100
sr = 16000
window_size = sr // 2
step_size = window_size // 2

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        full_path = os.path.join(folder_path, filename)
        base_filename, _ = os.path.splitext(filename)

        y, sr = load_and_filter_audio(full_path, sr)
        energy = calculate_energy(y, window_size, step_size)
        find_and_save_high_energy_segments(
            y, sr, energy, threshold, base_filename, window_size, output_folder
        )
