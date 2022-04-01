#!/usr/bin/env python3
import rospy
import math
import time
import moveit_commander
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import SwitchController

'''
Control actual robot
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.11.3
roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=false
roslaunch ur5_moveit_config moveit_rviz.launch config:=true
rosrun rqt_controller_manager rqt_controller_manager
'''

class targetTalker(object):
    def __init__(self):
        self.robot = moveit_commander.RobotCommander()
        print(self.robot.get_current_state())
        self.mani = moveit_commander.MoveGroupCommander("manipulator")
        self.pub_pos = rospy.Publisher('scaled_pos_traj_controller/command', JointTrajectory, queue_size=10)
        self.pub_vel = rospy.Publisher('joint_group_vel_controller/command', Float64MultiArray, queue_size=10)
        rospy.sleep(1)

    def setJointAngles(self, angle):
        self.mani.clear_pose_targets()
        self.mani.set_joint_value_target(angle)
        self.mani.go()

    def initPose(self):
        angle_init  = [math.pi/4, -math.pi/2, math.pi/2, -math.pi, -math.pi/2, 0]
        self.setJointAngles(angle_init)

    def readyPose(self):
        # angle_init  = [(95/180)*math.pi, (-185/180)*math.pi, (155/180)*math.pi, (-150/180)*math.pi, (-140/180)*math.pi, 0]
        self.changeControllerVel2Pos()
        self.moveWrist3(0)
        time.sleep(1)
        self.changeControllerPos2Vel()

    def moveWrist3(self, theta):
        self.theta = theta
        angle_init = [1.6579816341400146, -3.228877369557516, 2.7052814960479736, -2.617951218281881, -2.4435017744647425, self.theta]
        self.setJointAngles(angle_init)

    def changeControllerPos2Vel(self):
        rospy.wait_for_service("/controller_manager/switch_controller")
        try:
            switch_controller = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
            ret = switch_controller(['joint_group_vel_controller'], ['scaled_pos_traj_controller'], 2, False, 0.0)
        except rospy.ServiceException:
            print("Service call failed: %s")

    def changeControllerVel2Pos(self):
        rospy.wait_for_service("/controller_manager/switch_controller")
        try:
            switch_controller = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
            ret = switch_controller(['scaled_pos_traj_controller'], ['joint_group_vel_controller'], 2, False, 0.0)
        except rospy.ServiceException:
            print("Service call failed: %s")

    def commandWrist3_pos(self, theta):
        # ref from tracik
        tr0 = JointTrajectory()
        tr0.header.frame_id = "base_link"
        tr0.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        tr0.points = [JointTrajectoryPoint()]
        tr0.points[0].positions = [1.6579816341400146, -3.228877369557516, 2.7052814960479736, -2.617951218281881, -2.4435017744647425, theta]
        # tr0.points[0].velocities = [0, 0, 0, 0, 0, 0.5]
        tr0.header.stamp = rospy.Time.now()
        tr0.points[0].time_from_start = rospy.Duration(0.01)
        self.pub_pos.publish(tr0)

    def commandWrist3_vel(self, omega):
        msg = Float64MultiArray()
        msg.data = [0, 0, 0, 0, 0, omega]
        msg.layout.data_offset = 1
        self.pub_vel.publish(msg)

if __name__ == '__main__':
    rospy.init_node("IK_target_takler", disable_signals=True)
    rate = rospy.Rate(100) # 100Hz
    target = targetTalker()

    target.changeControllerVel2Pos()
    target.readyPose()
    time.sleep(1)
    target.changeControllerPos2Vel()

    """oscillator"""
    # A = 0.5
    # f = 0.1
    # t = 0.0
    # while not rospy.is_shutdown():
    #     theta = A*np.cos(2*np.pi*f*t)
    #     target.commandWrist3_vel(theta)
    #     t += 0.01
    #     print(theta)
    #     rate.sleep()

    """key input"""
    scale = 0.1
    tgt = 0.0
    while not rospy.is_shutdown():
        key = input('command: ')
        if key == 'a':
            tgt += 0.1*scale
        if key == 'd':
            tgt += -0.1*scale
        if key == 'state':
            print(target.mani.get_current_pose())
        if key == 'scale':
            key = raw_input('current is {}, change into ' .format(scale))
            scale = float(key)
        
        target.commandWrist3_vel(tgt)
