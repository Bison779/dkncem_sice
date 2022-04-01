#!/usr/bin/env python3
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""
---- comparison ----
DKN: nomral Deep Koopman Network
DelayDKN: DKN input is 50 steps history data (same as LSTM)
FCN: nomral Fully Connected Network
DelayFCN: FCN input is 50 steps history data (same as DelayDKN)
LSTM: input is 50 steps

---- state space----
th : angle of pendulim
dth: angular velocity of pendulum
q  : joint angle of robot
dq : joint velocity of robot
u  : control input (joint velocity)

---- data ----
column: [th, dth, q, dq, u]
row   : 20Hz*30sec = 600 samples
trial : 10 samples (0-4: from right side, 5-9: from left side)
"""


def load_reward_sum(filename):
    samples = 5     # num=10: full dataset, num=5: only first 5 trials (it means init pose is only from right side)
    sum_reward = []
    for i in range(5,10):
        reward, _, _ = load_pickle(filename+"_"+str(i))
        sum_reward.append(np.sum(reward))
    return sum_reward

def load_each_data(filename, num):
    reward, x, dx = load_pickle(filename+"_"+str(num))
    return reward, x, dx

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        reward = -(data[:,0]**2 + data[:,1]**2) # reward_function_weight = [1,1]
        return reward, data[:,0], data[:,1]


if __name__ == '__main__':
    filenames = ["FCN",
                 "DKN",
                 "DelayFCN",
                 "frozen_graph_soft_9eigfn"]

    labels = ["FCN",
              "DKN",
              "DelayFCN", 
              "DelayDKN"]

    # alphas = [0.4,0.4,0.4,1]
    alphas = [1,1,1,1]

    """  Sum of rewards """
    rewards = []
    for f in filenames:
        r = load_reward_sum(f)
        rewards.append(r)

    fig1, ax1 = plt.subplots()
    plt.boxplot(rewards)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("reward")


    """  State Space """
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(2, 1, 1)
    ax2 = fig2.add_subplot(2, 1, 2)

    for i in range(0,2):
        r, x, dx = load_each_data(filenames[i], num=5)    # num: set trial number 0-9
        ax1.plot(r[:600], label=labels[i], alpha=alphas[i])
    for i in range(2,4):
        r, x, dx = load_each_data(filenames[i], num=5)    # num: set trial number 0-9
        ax2.plot(r[:600], label=labels[i], alpha=alphas[i])

    ax1.set_xlim(0, 600)
    ax1.set_ylim(-1.8, 0.1)
    ax1.set_ylabel("reward")
    ax1.legend(loc='lower right')
    ax2.set_xlim(0, 600)
    ax2.set_ylim(-1.8, 0.1)
    ax2.set_xlabel("time step")
    ax2.set_ylabel("reward")
    ax2.legend(loc='lower right')


    plt.show()
