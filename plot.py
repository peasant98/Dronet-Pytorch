import numpy as np
import matplotlib.pyplot as plt

def plot(path):
    arr = np.loadtxt(path)
    return arr

if __name__ == '__main__':

    arr = plot('models/losses.txt')
    plt.plot(arr[::,0])
    plt.plot(arr[::,1])
    plt.show()
