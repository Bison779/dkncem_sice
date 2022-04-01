#!/usr/bin/env python3
import pickle
import rospy
import listener_mocap
import listener_wrist3
import talker_wrist3_target
import sys
import numpy as np
import copy
from std_msgs.msg import Float64MultiArray

"""
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.11.3 limited:=false
roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=false limited:=false
rosrun rqt_controller_manager rqt_controller_manager
rosrun plotjuggler plotjuggler
"""

if __name__ == '__main__':
    rospy.init_node("exp_commander", disable_signals=True)
    rate = rospy.Rate(8)
    pub = rospy.Publisher('my_data/val', Float64MultiArray, queue_size=10)
    msg = Float64MultiArray()

    ur5 = talker_wrist3_target.targetTalker()
    wrist3 = listener_wrist3.jointstateListener()
    mocap = listener_mocap.mocapPoseListener("P_2")

    ur5.readyPose()

    input('Press enter to start experiment')

    data = []
    x_tgt = 0.0
    th_tgt = 0.0

    """STANDUP
    P = 0.2
    D = 0.6
    I = 0.0

    if np.sign(np.tan(dx/x)) == -1:
        P = 0.1
        D = 0.5
        I = 0.1
    """

    """OSC
        P = 0.5
        D = 0.3
        I = 0.0
    """

    """69cm,under
    P = 0.2
    D = 0.6
    I = 0.0

    if np.sign(np.tan(dx/x)) == -1:
        P = 0.3
        D = 0.5
        I = 0.1
    """

    # while not rospy.is_shutdown():
    for i in range(8*60*50):
        print(i)

        t_start = rospy.get_time()
        x = mocap.theta
        dx = mocap.omega_smooth * 120
        th = wrist3.theta
        dth = wrist3.omega
        
        x_current = np.array([x,dx])
        x_target = np.array([0,0])

        P = 0.2
        D = 0.6
        I = 0.0

        u_PD = P*(x-x_tgt) + D*dx - I*(th-th_tgt)

        u = u_PD
        ur5.commandWrist3_vel(u)

        msg.data = [x, dx, th, dth, u]
        pub.publish(msg)

        data.append(x)
        data.append(dx)
        data.append(th)
        data.append(dth)
        data.append(u)

        rate.sleep()

        print(rospy.get_time() - t_start)


    # Finish
    ur5.commandWrist3_vel(0)
    col = 5
    savedata = np.array(data).reshape(int(len(data)/col),col)
    with open("../src/savedata","wb") as f:
        pickle.dump(savedata,f)

    sys.exit()