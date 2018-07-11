import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
import pyaudio

def create_wave(f0, fs):#振幅、周波数、サンプリング周波数

    binwave = []
    #binwave = create_sin(1,f0,fs)
    for i in range(10):
        binwave += create_sin((i+1)/10,f0,fs)
    
    binwave = struct.pack("h" * len(binwave), * binwave)

    w = wave.Wave_write("inputwave.wav")
    p = (1, 2, fs, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()
    print(len(binwave))

def create_sin(A,f0,fs):#振幅、周波数、サンプリング周波数
    point = np.arange(0,fs*20)
    sin_wave = A*np.sin(2*np.pi*f0*(point/fs)**3)
    sin_wave = [int(x * 32767.0) for x in sin_wave]
    return sin_wave
