import sys

import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

import soundfile as sf

if __name__ == '__main__':
    plt.close("all")

    filename = sys.argv[1]
    wav, fs = sf.read(filename)
    print(wav)
    totaltime = len(wav)/fs
    time = np.arange(0, totaltime, 1/fs)
    plt.plot(time,wav)
    plt.show()

"""
    wav_l = wav[:, 0]
    wav_r = wav[:, 1]
    fig = plt.figure(1, figsize=(8, 10))
    ax = fig.add_subplot(211)
    ax.plot(
"""
