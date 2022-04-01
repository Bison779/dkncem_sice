#!/usr/bin/env python3
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
---- comparison ----
DKN: nomral Deep Koopman Network
DelayDKN: DKN input is 50 steps history data (same as LSTM)
FCN: nomral Fully Connected Network
DelayFCN: FCN input is 50 steps history data (same as DelayDKN)
LSTM: input is 50 steps

---- state space----
th : angle of pendulim
dth: angular velocity of pendulum
q  : joint angle of robot
dq : joint velocity of robot
u  : control input (joint velocity)

---- data ----
column: [th, dth, q, dq, u]
row   : 20Hz*30sec = 600 samples
"""

class plotResults(object):
    def __init__(self, dir):
        self.fig1, self.ax1 = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        self.fig3, self.ax3 = plt.subplots()
        self.dir = dir
        self.i = 1

    def load_pickle(self,filename):
        with open('../src/results/'+self.dir+'/'+filename, 'rb') as f:
            data = pickle.load(f)
            loss = np.square(data[:,0:2])
            reward = 0-(50*loss[:,0] + 10*loss[:,1]) # reward_function_weight = [50,10]
            print(reward)
            self.ax1.plot(data[:,0], data[:,1], label=filename)
            self.ax2.plot(reward, label=filename)
            self.ax3.bar(self.i, np.sum(reward[0:]), label=filename)
            self.i += 1

    def show_ss(self):
        self.ax1.set_xlabel("th")
        self.ax1.set_ylabel("dth")
        self.ax1.legend()
        self.ax2.set_xlabel("step")
        self.ax2.set_ylabel("reward")
        self.ax2.legend()
        self.ax3.set_ylabel("reward")
        self.ax3.legend()
        plt.show()


if __name__ == '__main__':
    # dir = "20220315"
    # dir = "20220317_left"
    dir = "20220317_right"
    ss = plotResults(dir)

    ss.load_pickle("DKN")
    ss.load_pickle("FCN")
    ss.load_pickle("LSTM")
    ss.load_pickle("DelayDKN")
    ss.load_pickle("DelayFCN")

    ss.show_ss()

    sys.exit()