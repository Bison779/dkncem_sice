#!/usr/bin/env python3
from fn_control_prediction import fn_dynamics
from cem_vanilla import fn_control_prediction
from convert_keras_frozen import load_default_frozen_model
import listener_mocap
import listener_wrist3
import talker_wrist3_target
import tool_statedata
import rospy
import sys
import numpy as np
"""
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.11.3 limited:=false
roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=false limited:=false
rosrun rqt_controller_manager rqt_controller_manager
rosrun plotjuggler plotjuggler
"""

""" reward
DKN: [50,10]
FCN: [100,10]
"""

if __name__ == '__main__':
    rospy.init_node("exp_commander", disable_signals=True)
    ur5       = talker_wrist3_target.targetTalker()
    wrist3    = listener_wrist3.jointstateListener()
    mocap     = listener_mocap.mocapPoseListener("P_2")
    tool_real = tool_statedata.datacontroller("robot")
    tool_sim  = tool_statedata.datacontroller("simulation")


    # modelname = "old/DKN_Model10_easy"
    # modelname = "old/DKN_Model_11_V9_Combinations_2"
    # modelname = "old/FCN_Model_11_V9_Combinations_2"

    # modelname = "DKN_oldArch_E250"
    modelname = "DKN_newArch_E250"
    # modelname = "FCN_E25"

    model = load_default_frozen_model(modelname)

    freq = 20
    rate = rospy.Rate(freq)
    p_tgt = 0.0
    x_target = np.array([0,0])
    u_temp = np.zeros(3)

    for i in range(10):
        ur5.readyPose()
        input('trial {}: Press enter to start experiment' .format(i))

        while not rospy.is_shutdown():
        # for j in range(freq*30):
            t_start = rospy.get_time()
            p = mocap.theta
            dp = mocap.omega_smooth * 120
            th = wrist3.theta
            dth = wrist3.omega
            
            x_current = np.array([p,dp,th])
            u_next_step, reward = fn_control_prediction(model, x_current, x_target)
            u_temp = np.append(u_temp, u_next_step)
            u_temp = np.delete(u_temp, 0)
            u_smooth = np.mean(u_temp)

            P = 0.1
            D = 0.3
            u_PD = P*(p-p_tgt) + D*dp

            # u = u_PD
            u = u_next_step
            # u = u_smooth

            # x_pred = fn_dynamics(model, np.expand_dims(x_current,axis=0),u)[0]

            ur5.commandWrist3_vel(u)
            tool_real.dataAppend(p, dp, th, dth, u)
            # tool_sim.dataAppend(x_pred[0], x_pred[1], th, dth, u_PD)
            rate.sleep()
            print(rospy.get_time() - t_start)

        ur5.commandWrist3_vel(0)
        tool_real.detaSave(modelname+"_"+str(i))

    sys.exit()