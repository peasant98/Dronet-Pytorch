import numpy as np
import matplotlib.pyplot as plt

def plot(path):
    arr = np.loadtxt(path)
    return arr

if __name__ == '__main__':

    arr = plot('models/losses.txt')
    plt.plot(arr[:,0], label='Training Loss')
    plt.plot(arr[:,1], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()
