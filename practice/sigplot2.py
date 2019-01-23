import sys

import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

import soundfile as sf

if __name__ == '__main__':
    plt.close("all")

    filename = sys.argv[1]
    wav, fs = sf.read(filename)
    totaltime = len(wav)/fs
    time = np.arange(0, totaltime, 1/fs)
    plt.plot(time,wav, label="Real")

    filename = sys.argv[2]
    wav, fs = sf.read(filename)
    totaltime = len(wav)/fs
    time = np.arange(0, totaltime, 1/fs)
    plt.plot(time,wav, label="NN")

    plt.xlabel("Times [s]")
    plt.ylabel("Amplitude [arb. unit]")
    plt.legend()
    plt.xlim([0,0.02])
    plt.show()
