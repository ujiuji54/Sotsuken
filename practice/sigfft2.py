import sys

import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf

def fft(filename):
    wav, fs = sf.read(filename)
    print(wav)
    totaltime = len(wav)/fs
    time = np.arange(0, totaltime, 1/fs)

    d=1.0/fs
    size = 2048
    s = 10000
    fftwav=np.abs(np.fft.fft(wav[s:s+size]))
    frq=np.fft.fftfreq(size,d)
    #plt.yscale("log")
    Max=max(fftwav)
    dB=20*np.log10(fftwav/Max)
    return frq,dB

if __name__ == '__main__':
    plt.close("all")

    filename = sys.argv[1]
    frq,dB=fft(filename)
    plt.plot(frq[0:int(len(frq)/2)-1],dB[0:int(len(dB)/2)-1],label="Teacher")

    filename = sys.argv[2]
    frq,dB=fft(filename)
    plt.plot(frq[0:int(len(frq)/2)-1],dB[0:int(len(dB)/2)-1],label="Model")

    plt.axis([0,10000,min(dB)-20,5])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.legend()

    plt.show()
