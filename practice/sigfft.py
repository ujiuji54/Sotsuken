import sys

import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf

if __name__ == '__main__':
    plt.close("all")

    filename = sys.argv[1]
    wav, fs = sf.read(filename)
    print(wav)
    totaltime = len(wav)/fs
    time = np.arange(0, totaltime, 1/fs)

    d=1.0/fs
    size = 2048
    s = 10000
    fftwav=np.fft.fft(wav[s:s+size])
    frq=np.fft.fftfreq(size,d)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(frq,np.abs(fftwav))
    plt.axis([0,10000,0,max(np.abs(fftwav))])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [arb. unit]")

    plt.show()
