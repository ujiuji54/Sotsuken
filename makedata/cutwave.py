from argparse import ArgumentParser
import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
import pyaudio

def cut_wave():
    args = parse_args()
    print("output_filename:")
    output_filename = input()

    iw = wave.open(args.input_file,"r")
    print(iw.getnchannels())
    data = np.frombuffer(iw.readframes(iw.getnframes()),dtype=np.int16)
    print(data)
    print(len(data))

    for i in range(10):
        binwave = data[i*240000:(i+1)*240000]
        ow = wave.Wave_write(output_filename+str(i)+".wav")
        ow.setparams((1, 2, 48000, len(binwave), 'NONE', 'not compressed'))
        ow.writeframes(binwave)
        ow.close()
    iw.close()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file", "-i",
        help="input file (*.wav)")
    return parser.parse_args()

if __name__=="__main__":
    cut_wave()
