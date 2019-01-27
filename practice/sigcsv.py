import sys

import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf

if __name__ == '__main__':
    plt.close("all")

    filename = sys.argv[1]
    x_data = np.loadtxt(filename,delimiter=",",skiprows=1,usecols=1)
    y_data = np.loadtxt(filename,delimiter=",",skiprows=1,usecols=2)

    plt.plot(x_data,y_data)
    plt.xlabel("Loss")
    plt.ylabel("Epochs")
    plt.show()
