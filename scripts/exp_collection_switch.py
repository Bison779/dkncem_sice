#!/usr/bin/env python3
import rospy
import listener_mocap
import listener_wrist3
import talker_wrist3_target
import tool_statedata
import sys
import numpy as np

"""
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.11.3 limited:=false
roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=false limited:=false
rosrun rqt_controller_manager rqt_controller_manager
rosrun plotjuggler plotjuggler
"""

# under 58cm
"""OSC_1
P = 0.3
D = 0.1
"""
"""OSC_2
P = 0.3
D = 0.2
"""
"""STANDUP_1
P = 0.1
D = 0.2
"""
"""STANDUP_1
P = 0.1
D = 0.3
"""

if __name__ == '__main__':
    rospy.init_node("exp_commander", disable_signals=True)
    ur5 = talker_wrist3_target.targetTalker()
    wrist3 = listener_wrist3.jointstateListener()
    mocap = listener_mocap.mocapPoseListener("P_2")
    tool = tool_statedata.datacontroller("robot")

    freq = 20
    rate = rospy.Rate(freq)
    p_tgt = 0.0
    side = False

    ur5.readyPose()

    input('Press enter to start experiment')

    for i in range(30):
        
        p_tgt = 0.0
        for j in range(freq*30):
            t_start = rospy.get_time()
            p = mocap.theta
            dp = mocap.omega_smooth * 120
            th = wrist3.theta
            dth = wrist3.omega

            P = 0.1
            D = 0.3

            u_PD = P*(p-p_tgt) + D*dp
            # rand = noise*np.random.randn()
            u = u_PD

            ur5.commandWrist3_vel(u)
            tool.dataAppend(p, dp, th, dth, u)
            rate.sleep()
            print(i, p_tgt, rospy.get_time() - t_start)

        if side==True:
            p_tgt = -0.1
            side = False
            for j in range(freq*30):
                t_start = rospy.get_time()
                p = mocap.theta
                dp = mocap.omega_smooth * 120
                th = wrist3.theta
                dth = wrist3.omega

                # P = 0.1
                # D = 0.3

                u_PD = P*(p-p_tgt) + D*dp
                u = u_PD

                ur5.commandWrist3_vel(u)
                tool.dataAppend(p, dp, th, dth, u)
                rate.sleep()
                print(i, p_tgt, rospy.get_time() - t_start)
        else:
            p_tgt = 0.1
            side = True
            for j in range(freq*30):
                t_start = rospy.get_time()
                p = mocap.theta
                dp = mocap.omega_smooth * 120
                th = wrist3.theta
                dth = wrist3.omega

                # P = 0.1
                # D = 0.3

                u_PD = P*(p-p_tgt) + D*dp
                u = u_PD

                ur5.commandWrist3_vel(u)
                tool.dataAppend(p, dp, th, dth, u)
                rate.sleep()
                print(i, p_tgt, rospy.get_time() - t_start)

        # for j in range(freq*20):
        #     t_start = rospy.get_time()
        #     p = mocap.theta
        #     dp = mocap.omega_smooth * 120
        #     th = wrist3.theta
        #     dth = wrist3.omega

        #     ur5.commandWrist3_vel(-th)
        #     tool.dataAppend(p, dp, th, dth, u)
        #     rate.sleep()
        #     print(rospy.get_time() - t_start)

    # Finish
    tool.detaSave()
    sys.exit()
