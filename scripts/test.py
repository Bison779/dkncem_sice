from fn_control_prediction import *
from convert_keras_frozen import *
import numpy as np
import time

"""
kill -9 $(lsof -t /dev/nvidia*)
"""

def test():
	model = load_default_frozen_model()
	
	x_current = np.array([0,0])
	x_target = np.array([np.pi,0])
	
	u_next_step = fn_control_prediction(model,x_current,x_target)
	
	print(u_next_step)


if __name__ == '__main__':
	modelname = "DKN_oldArch_E250"
	convert_keras_frozen(modelname)