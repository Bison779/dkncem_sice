#!/usr/bin/env python3
from fn_control_prediction import fn_dynamics
from cem_vanilla import fn_control_prediction
from convert_keras_frozen import load_default_frozen_model
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

if __name__ == '__main__':
    rospy.init_node("exp_simulation", disable_signals=True)
    tool_sim  = tool_statedata.datacontroller("simulation")

    # modelname = "Model_8-2_E25"
    # modelname = "Model_8-2_E25_noright"
    # modelname = "Model_8_V6_Combinations_VV1"
    # modelname = "Model_9_beamwall"
    # modelname = "Model10_easy"
    # modelname = "Model10-4_large_noresults"
    # modelname = "Model10-2_result_2"
    # modelname = "Model10-3_result_2_3"
    # modelname = "LSTM_V9_Combinations_2"

    # modelname = "Model_11_V9_Combinations_2"
    modelname = "FCN_Model_11_V9_Combinations_2"
    model = load_default_frozen_model(modelname)

    
    freq = 20
    p_tgt = 0.0
    x_target = np.array([0,0])
    x_pred = [-1.0, 0, -0.5]
    u = 0

    for i in range(freq*10):
        t_start = rospy.get_time()
        p = x_pred[0]
        dp = x_pred[1]
        th = x_pred[2]
        dth = u
        
        x_current = np.array([p,dp,th])
        u_next_step, reward = fn_control_prediction(model, x_current, x_target)

        P = 0.1
        D = 0.3
        u_PD = P*(p-p_tgt) + D*dp

        # u = u_PD
        u = u_next_step

        x_pred = fn_dynamics(model, np.expand_dims(x_history,axis=0))[0]

        tool_sim.dataAppend(p, dp, th, dth, u)
        print(rospy.get_time() - t_start)


    tool_sim.dataPlot()
    # tool_sim.detaSave("result_sim")
    sys.exit()