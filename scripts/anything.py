#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def changeData(filename):
    with open(filename, 'rb') as f:
        raw = pickle.load(f)
        print(np.shape(raw))
    for i in range(10):
        with open("../"+filename+"_"+str(i),"wb") as f:
            data = raw[600*i:600*(i+1),:]
            pickle.dump(data,f)

if __name__ == '__main__':
    changeData("LSTM")
