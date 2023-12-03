import os
import librosa
import soundfile as sf
from tqdm import tqdm

input_folder = "/home/yifeng/SVS/data/violin"
output_folder = "/home/yifeng/SVS/data/violin_segments"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

segment_length = 4
target_sr = 16000

for file_name in tqdm(os.listdir(input_folder)):
    if file_name.endswith(".wav") and "normal" in file_name:
        file_path = os.path.join(input_folder, file_name)
        y, original_sr = librosa.load(file_path, sr=None)

        if original_sr != target_sr:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)

        samples_per_segment = segment_length * target_sr

        num_segments = int(len(y) / samples_per_segment)

        for i in range(num_segments):
            start_sample = i * samples_per_segment
            end_sample = start_sample + samples_per_segment

            segment = y[start_sample:end_sample]
            segment_file_name = f"{os.path.splitext(file_name)[0]}_{i}.wav"
            segment_file_path = os.path.join(output_folder, segment_file_name)

            sf.write(segment_file_path, segment, target_sr)
