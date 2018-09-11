import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
import pyaudio
from scipy import fromstring, int16

def cut_wave():
    iw = wave.open("outputwave.wav","r")
    data = iw.readframes(iw.getnframes())
    data = fromstring(data, dtype=int16)
    print(data)
    print(len(data))

    for i in range(10):
        binwave = data[i*607200:(i+1)*607200]
        binwave = struct.pack("h" * len(binwave), * binwave)
        ow = wave.Wave_write("train_y"+str(i)+".wav")
        p = (1, 2, 48000, len(binwave), 'NONE', 'not compressed')
        ow.setparams(p)
        ow.writeframes(binwave)
        ow.close()
    iw.close()
