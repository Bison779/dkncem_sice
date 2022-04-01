#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose

class mocapPoseListener(object):
    def __init__(self, bodyname):
        topic = 'mocap_pose_topic/' + bodyname + '_pose'
        rospy.Subscriber(topic, PoseStamped, self.callback)
        self.pose = Pose()
        self.theta        = 0.0
        self.omega        = 0.0
        self.omega_smooth = 0.0
        self.r            = 0.0
        self.theta_pre    = 0.0
        moving            = 5
        self.omega_temp = np.zeros(moving)
        
    def callback(self, data):
        self.pose  = data.pose
        self.theta = np.arctan2(self.pose.position.x, self.pose.position.y)
        self.r     = np.sqrt(self.pose.position.x**2 + self.pose.position.y**2)
        self.omega = self.theta - self.theta_pre
        omega_temp = np.append(self.omega_temp, self.omega)
        self.omega_temp = np.delete(omega_temp, 0)
        self.omega_smooth = np.mean(self.omega_temp)
        self.theta_pre = self.theta

if __name__ == '__main__':
    rospy.init_node("mocap_listener", disable_signals=True)
    mocap = mocapPoseListener("P_2")

    while not rospy.is_shutdown():
        print(mocap.theta)