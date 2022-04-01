#!/usr/bin/env python3
import rospy
import numpy as np
import pickle
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class datacontroller(object):
    def __init__(self, topic):
        self.pub = rospy.Publisher('my_data/'+topic, Float64MultiArray, queue_size=10)
        self.data = []
        
    def dataAppend(self, p, dp, th, dth, u):
        self.data.append(p)
        self.data.append(dp)
        self.data.append(th)
        self.data.append(dth)
        self.data.append(u)

        msg = Float64MultiArray()
        msg.data = [p, dp, th, dth, u]
        self.pub.publish(msg)

    def detaSave(self, filename="savedata"):
        col = 5
        savedata = np.array(self.data).reshape(int(len(self.data)/col),col)
        with open("../src/"+filename,"wb") as f:
            pickle.dump(savedata,f)
        self.data = []

    def dataPlot(self):
        col = 5
        plotdata = np.array(self.data).reshape(int(len(self.data)/col),col)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(plotdata[:,0], plotdata[:,1], plotdata[:,2], s=2, cmap='jet')
        ax.set_xlabel("p")
        ax.set_ylabel("dp")
        ax.set_zlabel("theta")
        plt.show()