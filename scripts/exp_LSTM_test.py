#!/usr/bin/env python3
from cem_LSTM import *
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

""" reward (new notebook)
horizon_num: 20
DelayDKN: [50,10]
"""

""" reward (old)
LSTM: [50,10]
DelayDKN: [30,10]
DelayFCN: [100,0]
"""

if __name__ == '__main__':
    rospy.init_node("exp_commander", disable_signals=True)
    ur5       = talker_wrist3_target.targetTalker()
    wrist3    = listener_wrist3.jointstateListener()
    mocap     = listener_mocap.mocapPoseListener("P_2")
    tool_real = tool_statedata.datacontroller("robot")
    tool_sim  = tool_statedata.datacontroller("simulation")

    modelname = "old/LSTM_V9_Combinations_2"
    # modelname = "old/DelayDKN_Model_12_V9_Combinations_DelayEmbedding"
    # modelname = "old/DelayDKN_Model_13_V9_Combinations_DelayEmbedding_more_epochs"
    # modelname = "old/DelayFCN_Model_12_V9_fixed_E250"
    # modelname = "old/DelayFCN_Model_12_V9_fixed_E25"

    # modelname = "latent_analysis/new_DelayDKN_soft_1eigfn"
    # modelname = "latent_analysis/new_DelayDKN_soft_2eigfn"
    # modelname = "latent_analysis/new_DelayDKN_soft_3eigfn"
    # modelname = "latent_analysis/new_DelayDKN_soft_4eigfn"
    # modelname = "latent_analysis/new_DelayDKN_soft_5eigfn"
    # modelname = "latent_analysis/new_DelayDKN_soft_10eigfn"

    # modelname = "DelayDKN_oldArch_E250"
    # modelname = "DelayDKN_newArch_E250"
    # modelname = "DelayFCN_E25"

    # modelname = "frozen_graph_standup_1_1ceigfn_0reigfn"
    # modelname = "frozen_graph_standup_1_2ceigfn_0reigfn"
    # modelname = "frozen_graph_standup_1_10ceigfn_0reigfn"

    # modelname = "frozen_graph_soft_1eigfn"
    # modelname = "frozen_graph_soft_2eigfn"
    # modelname = "frozen_graph_soft_3eigfn"
    # modelname = "frozen_graph_soft_4eigfn"
    # modelname = "frozen_graph_soft_5eigfn"
    # modelname = "frozen_graph_soft_6eigfn"
    # modelname = "frozen_graph_soft_7eigfn"
    # modelname = "frozen_graph_soft_8eigfn"
    modelname = "frozen_graph_soft_9eigfn"
    # modelname = "frozen_graph_soft_10eigfn"
    # modelname = "frozen_graph_soft_11eigfn"
    # modelname = "frozen_graph_soft_20eigfn"
    # modelname = "frozen_graph_soft_30eigfn"
    # modelname = "frozen_graph_soft_50eigfn"
    # modelname = "frozen_graph_soft_1eigfn_largernetwork"

    model = load_default_frozen_model(modelname)

    freq = 20
    rate = rospy.Rate(freq)
    p_tgt = 0.0
    x_target = np.array([0,0])
    u_temp = np.zeros(3)
    history = 50
    x_0 = np.array([mocap.theta, 0, wrist3.theta, 0])
    x_history = np.repeat(x_0[np.newaxis],history-1,axis=0)

    ur5.readyPose()
    input('Press enter to start experiment')

    for i in range(10):

        input('Press enter to start experiment')

        # for j in range(freq*10):
        #     t_start = rospy.get_time()
        #     p = mocap.theta
        #     dp = mocap.omega_smooth * 120
        #     th = wrist3.theta
        #     dth = wrist3.omega
            
        #     x_current = np.array([p,dp,th,0])
        #     x_history = np.concatenate([x_history,np.expand_dims(x_current,axis=0)], axis=0)
        #     u_next_step, reward = fn_control_prediction(model, x_history, x_target)

        #     u_temp = np.append(u_temp, u_next_step)
        #     u_temp = np.delete(u_temp, 0)
        #     u_smooth = np.mean(u_temp)

        #     P = 0.1
        #     D = 0.3
        #     u_PD = P*(p-p_tgt) + D*dp

        #     u = u_PD
        #     # u = u_next_step
        #     # u = u_smooth

        #     x_pred = fn_dynamics(model, np.expand_dims(x_history,axis=0))[0]

        #     ur5.commandWrist3_vel(u)
        #     tool_real.dataAppend(p, dp, th, dth, u)
        #     tool_sim.dataAppend(x_pred[0], x_pred[1], th, dth, u_PD)
            
        #     x_history = np.delete(x_history, 0, 0)                
        #     x_history[-1,3] = u

        #     rate.sleep()
        #     print(rospy.get_time() - t_start)

        # while not rospy.is_shutdown():
        for j in range(freq*30):
            t_start = rospy.get_time()
            p = mocap.theta
            dp = mocap.omega_smooth * 120
            th = wrist3.theta
            dth = wrist3.omega
            
            x_current = np.array([p,dp,th,0])
            x_history = np.concatenate([x_history,np.expand_dims(x_current,axis=0)], axis=0)
            u_next_step, reward = fn_control_prediction(model, x_history, x_target)

            u_temp = np.append(u_temp, u_next_step)
            u_temp = np.delete(u_temp, 0)
            u_smooth = np.mean(u_temp)

            P = 0.1
            D = 0.2
            u_PD = P*(p-p_tgt) + D*dp

            # u = u_PD
            u = u_next_step
            # u = u_smooth

            x_pred = fn_dynamics(model, np.expand_dims(x_history,axis=0))[0]

            ur5.commandWrist3_vel(u)
            tool_real.dataAppend(p, dp, th, dth, u)
            tool_sim.dataAppend(x_pred[0], x_pred[1], th, dth, u_PD)
            
            x_history = np.delete(x_history, 0, 0)                
            x_history[-1,3] = u

            rate.sleep()
            print(rospy.get_time() - t_start)


        ur5.commandWrist3_vel(0)
        ur5.readyPose()
        tool_real.detaSave("results/"+modelname+"_"+str(i))
        # rospy.sleep(10)

    sys.exit()