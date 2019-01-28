import sys

import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf

if __name__ == '__main__':
    plt.close("all")

    filename = sys.argv[1]
    x_data1 = np.loadtxt(filename,delimiter=",",skiprows=1,usecols=1)
    y_data1 = np.loadtxt(filename,delimiter=",",skiprows=1,usecols=2)

    filename = sys.argv[2]
    x_data2 = np.loadtxt(filename,delimiter=",",skiprows=1,usecols=1)
    y_data2 = np.loadtxt(filename,delimiter=",",skiprows=1,usecols=2)

    plt.plot(x_data1,y_data1,label="loss")
    plt.plot(x_data2,y_data2,label="valloss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
