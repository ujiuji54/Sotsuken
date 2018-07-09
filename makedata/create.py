import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
import pyaudio

def create_wave(A, f0, fs, t):#振幅、周波数、サンプリング周波数、秒数

    point = np.arange(0,fs*t)
    sin_wave = A*np.sin(2*np.pi*f0*(point/fs)*(point/fs))

    sin_wave = [int(x * 32767.0) for x in sin_wave]
    
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)

    w = wave.Wave_write("somewave.wav")
    p = (1, 2, fs, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()
    print(len(sin_wave))
