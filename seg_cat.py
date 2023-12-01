import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf


def load_audio(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr


def calculate_energy(y, window_size, step_size):
    energy = []
    for i in range(0, len(y) - window_size, step_size):
        window_energy = np.sum(np.square(y[i : i + window_size]))
        energy.append((i, window_energy))
    return energy


def find_and_save_high_energy_segments(
    y, sr, energy, threshold, base_filename, window_size
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
                print(segment_filename)
                sf.write(segment_filename, segment_audio, sr)
                segment_index += 1

                current_segment_start = None
                current_segment_end = None

    if current_segment_start is not None:
        segment_audio = y[current_segment_start:current_segment_end]
        segment_filename = f"{base_filename}_seg_{segment_index}.wav"
        print(segment_filename)
        sf.write(segment_filename, segment_audio, sr)


sr = 16000
filename = "/home/yyu479/ddsp-pytorch/data/cats/train/cat_2.wav"
threshold = 100
window_size = 3 * sr
step_size = window_size // 2
save_path = "spectrogram.png"
base_filename, _ = os.path.splitext(filename)

y, sr = load_audio(filename)
energy = calculate_energy(y, window_size, step_size)
find_and_save_high_energy_segments(y, sr, energy, threshold, base_filename, window_size)
