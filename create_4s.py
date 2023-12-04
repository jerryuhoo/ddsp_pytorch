import librosa
import numpy as np
import os
import soundfile as sf
import random


def load_audio_files(folder_path):
    audio_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            full_path = os.path.join(folder_path, filename)
            y, sr = librosa.load(full_path, sr=None)
            audio_files.append((y, sr))
    return audio_files


def combine_audio_files(audio_files, target_length=4.0):
    random.shuffle(audio_files)  # 随机打乱音频文件
    combinations = []
    current_combination = []
    current_length = 0.0

    for y, sr in audio_files:
        duration = len(y) / sr
        if current_length + duration <= target_length:
            current_combination.append((y, sr))
            current_length += duration
        else:
            # 填充静音以达到目标长度
            if current_length < target_length:
                silence_length = int((target_length - current_length) * sr)
                silence = np.zeros(silence_length)
                current_combination.append((silence, sr))
            combinations.append(current_combination)
            current_combination = [(y, sr)]
            current_length = duration

    # 处理最后一个组合
    if current_combination:
        if current_length < target_length:
            silence_length = int((target_length - current_length) * sr)
            silence = np.zeros(silence_length)
            current_combination.append((silence, sr))
        combinations.append(current_combination)

    return combinations


def save_combined_audio(combinations, output_folder):
    for index, combination in enumerate(combinations):
        combined_audio = np.array([])
        sr = None

        for y, current_sr in combination:
            combined_audio = np.concatenate((combined_audio, y))
            sr = current_sr

        output_filename = f"combined_{index}.wav"
        output_file_path = os.path.join(output_folder, output_filename)
        sf.write(output_file_path, combined_audio, sr)


# 指定输入和输出文件夹
input_folder = "/home/yifeng/SVS/ddsp_pytorch/data/cats/train_seg"
output_folder = "/home/yifeng/SVS/ddsp_pytorch/data/cats/train2"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

audio_files = load_audio_files(input_folder)
combinations = combine_audio_files(audio_files)
save_combined_audio(combinations, output_folder)
