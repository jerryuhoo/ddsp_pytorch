import os
from struct import pack
from wave import open
import sys
import numpy as np
import AutoTune
from tqdm import tqdm

# Constants
FORM_CORR = 0
SCALE_ROTATE = 0
LFO_QUANT = 0
CONCERT_A = 440.0
FIXED_PITCH = 2.0
FIXED_PULL = 0.1
CORR_STR = 1.0
CORR_SMOOTH = 0.0
PITCH_SHIFT = 0.0  # This will be changed dynamically
LFO_DEPTH = 0.1
LFO_RATE = 1.0
LFO_SHAPE = 0.0
LFO_SYMM = 0.0
FORM_WARP = 0.0
MIX = 1.0
KEY = "c".encode()
CHUNK = 4096


def process_file(input_file, output_file, pitch_shift):
    global CHUNK
    wf = open(input_file, "rb")

    # If Stereo
    if wf.getnchannels() == 2:
        print(f"{input_file} is stereo, only mono files are supported.")
        return

    signal = wf.readframes(-1)
    fs = wf.getframerate()
    scale = 1 << 15
    intsignal = np.frombuffer(signal, dtype=np.int16)
    floatsignal = np.float32(intsignal) / scale

    fout = open(output_file, "w")
    fout.setnchannels(1)  # Mono
    fout.setsampwidth(2)  # Sample is 2 Bytes (2) if int16 = short int
    fout.setframerate(fs)  # Sampling Frequency
    fout.setcomptype("NONE", "Not Compressed")

    for i in range(0, len(floatsignal), CHUNK):
        SignalChunk = floatsignal[i : i + CHUNK]
        if i + CHUNK > len(floatsignal):
            CHUNK = len(SignalChunk)
        rawfromC = AutoTune.Tuner(
            SignalChunk,
            fs,
            CHUNK,
            SCALE_ROTATE,
            LFO_QUANT,
            FORM_CORR,
            CONCERT_A,
            FIXED_PITCH,
            FIXED_PULL,
            CORR_STR,
            CORR_SMOOTH,
            pitch_shift,
            LFO_DEPTH,
            LFO_RATE,
            LFO_SHAPE,
            LFO_SYMM,
            FORM_WARP,
            MIX,
            KEY,
        )
        shortIntvalues = np.int16(np.asarray(rawfromC) * (scale))
        outdata = pack("%dh" % len(shortIntvalues), *(shortIntvalues))
        fout.writeframesraw(outdata)

    fout.close()


def main(input_folder):
    skip_files = ["cat_14.wav"]
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".wav") and filename not in skip_files:
            input_file = os.path.join(input_folder, filename)
            for pitch_shift in range(-5, 6):  # From -5 to +5
                output_file = os.path.splitext(input_file)[0] + f"_{pitch_shift}.wav"
                print(f"Processing {input_file} with pitch shift {pitch_shift}")
                try:
                    process_file(input_file, output_file, pitch_shift)
                except Exception as e:
                    print(e)
                    print(
                        f"Failed to process {input_file} with pitch shift {pitch_shift}"
                    )
                    continue


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python TuneAndSaveToFile_folder.py <Input Folder>")
        sys.exit(0)

    input_folder = sys.argv[1]
    main(input_folder)
