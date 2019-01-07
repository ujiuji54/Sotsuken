# -*- coding: utf-8 -*-
from scipy import arange, cumsum, sin, linspace
from scipy import pi as mpi

# ==========================
#  時間変化するサイン波を作る
# ==========================
def make_time_varying_sine(start_freq, end_freq, A, fs, sec = 5.):
    freqs = linspace(start_freq, end_freq, num = int(round(fs * sec)))
    ### 角周波数の変化量
    phazes_diff = 2. * mpi * freqs / fs
    ### 位相
    phazes = cumsum(phazes_diff)
    ### サイン波合成
    ret = A * sin(phazes)

    return ret


from scikits.audiolab import wavwrite
A_default = 0.5
fs_default = 48000
sec = 30.
def test2():
    sine_wave = make_time_varying_sine(10, 20000, A_default, fs_default, sec)

    wavwrite(sine_wave, "../wav/time_varying_sine_wave.wav", fs = fs_default)


if __name__ == "__main__":
    test2()
