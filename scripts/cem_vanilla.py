import tensorflow as tf
import numpy as np
from convert_keras_frozen import load_default_frozen_model

# Predict next step control
def fn_control_prediction(frozen_func,xc,xt):

	# CEM parameters
	delta_time  = 0.25
	trials      = 5
	horizon_num = 10   # Predictive horizon for CEM
	sample_num  = 500 # Number of random control samples
	elite_limit = 5   # Number of elite control samples to return
	epsilon     = 0.1
	alpha_mean  = 0.1
	alpha_var   = 0.7
	
	# Load the (frozen) model
	#frozen_func = load_default_frozen_model()
	
	# Compute the optimal next step control using CEM
	return cross_entropy_method(frozen_func,xc,xt,delta_time=delta_time,trials=trials,horizon_num=horizon_num,sample_num=sample_num,
                               elite_limit=elite_limit,epsilon=epsilon,alpha_mean=alpha_mean,alpha_var=alpha_var)

# Predict next step dynamics
def fn_dynamics(model,x,u):
    x_aug = np.expand_dims(np.concatenate([x,np.expand_dims(u,axis=1)],axis=1),axis=1)
    xt = tf.convert_to_tensor(x_aug,dtype=float)
    dx = model(xt)[0]
    dx = dx.numpy()
    return dx[:,0,:3]
    
# Cross entropy method: Finds a control signal to move the starting state to the goal state, by minimising 
# the step-wise cross entropy
def cross_entropy_method(model,
                        current_state,
                        target_state,
                        delta_time=0.2,
                        trials=50,
                        horizon_num=100,
                        sample_num=100,
                        elite_limit=10,
                        epsilon=0.1,
                        alpha_mean=0.1,
                        alpha_var=0.7):
     
	rand_center = 0
	lim = 0.5
	min_lim = rand_center-lim
	max_lim = rand_center+lim

	for trial in range(trials):
		u_samples = np.random.uniform(low=min_lim,
				          high=max_lim,
				          size=(sample_num, horizon_num))
		var_u = np.var(u_samples[:,0])
		mean_u = np.mean(u_samples[:,0])
		reward = np.zeros(sample_num)

		horizon_state = np.repeat([current_state], sample_num, axis=0)
	    
		for horizon_step in range(horizon_num):
			horizon_state = fn_dynamics(model,horizon_state, u_samples[:,horizon_step])
			reward_function_weight = [100,10]

			reward = reward - (reward_function_weight[0]*(target_state[0] - horizon_state[:,0])**2 + reward_function_weight[1]*(target_state[1]-horizon_state[:,1])**2)
		    # reward = reward - (reward_function_weight[0]*np.abs(target_state[0] - horizon_state[:,0]) + reward_function_weight[1]*np.abs(target_state[1]-horizon_state[:,1]))
			# reward = reward + 1/(reward_function_weight[0]*abs(target_state[0]-abs(horizon_state[:,0])) + reward_function_weight[1]*abs(target_state[1]-abs(horizon_state[:,1])))
				
		
		reward_sort_idx = reward.argsort()
		sorted_u_samples = u_samples[reward_sort_idx[::-1]]
		sorted_rewards = reward[reward_sort_idx[::-1]]
	    
		elite_u = sorted_u_samples[:elite_limit,:]
		elite_r = sorted_rewards[:elite_limit]
	    
		var_u = (alpha_var*var_u)+((1-alpha_var)*np.var(elite_u[:,0]))
		mean_u = (alpha_mean*mean_u)+((1-alpha_mean)*np.mean(elite_u[:,0]))
	    
		min_lim = mean_u-var_u
		max_lim = mean_u+var_u
	    
		if var_u < epsilon:
			print(trial)
			break	
	return elite_u[0,0], elite_r[0]
