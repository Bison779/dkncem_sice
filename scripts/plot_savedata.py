#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class plotStateSpace(object):
    def __init__(self, dir):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.dir = dir

    def load_pickle(self,filename):
        with open('../src/'+self.dir+'/'+filename, 'rb') as f:
            data = pickle.load(f)
            self.plot_ss(data[:,0], data[:,1], data[:,2])

    def plot_ss(self,x,y,z):
        self.ax.plot(x,y,z)
        # self.ax.scatter(x,y,z, s=1)

    def show_ss(self):
        self.ax.set_xlabel("p")
        self.ax.set_ylabel("dp")
        self.ax.set_zlabel("theta")
        plt.show()


if __name__ == '__main__':
    
    dir = "20220309_easy"
    ss = plotStateSpace(dir)

    ss.load_pickle("standup_1")
    ss.load_pickle("standup_2")
    ss.load_pickle("osc_1")
    ss.load_pickle("osc_2")

    # ss.load_pickle("result_1")
    # ss.load_pickle("result_2")
    ss.load_pickle("result_3")

    ss.show_ss()