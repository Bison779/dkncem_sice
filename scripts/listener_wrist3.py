#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState

class jointstateListener(object):
    def __init__(self):
        topic = 'joint_states'
        rospy.Subscriber(topic, JointState, self.callback)
        self.wrist3 = 0.0
        
    def callback(self, data):
        self.theta = data.position[5]
        self.omega = data.velocity[5]


if __name__ == '__main__':
    rospy.init_node("jointstate_listener", disable_signals=True)
    wrist3 = jointstateListener()

    while not rospy.is_shutdown():
        print(wrist3)