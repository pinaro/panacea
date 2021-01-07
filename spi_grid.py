# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:52:39 2017

@author: Pinar Ozisik
@description: 
"""

from grid import *
from constants import *
from grid import coordinate_to_index
import numpy as np
import pickle

# Application of different CIs
def CH(estimates, delta, b):
    n = len(estimates)
    tmp = np.log(1/delta) / (2*n)
    tmp = b * np.sqrt(tmp)
    diff = np.mean(estimates) - tmp
    return diff

# Loop over all states, all actions such that IS weight is minimized
def min_weight(pi_e, pi_b):
    actions = len(ACTIONS)
    min_state = -1
    min_action = -1
    min_ratio = 1e+10
    for x in range(0, ROW):
        for y in range(0, COL):
            ind = coordinate_to_index(x, y)
            for a in range(0, actions): 
                ratio = pi_e[ind, a] / pi_b[ind, a]
                if(ratio < min_ratio):
                    min_ratio = ratio
                    min_state = (x, y)
                    min_action = a
    assert min_ratio < 1e+10 and min_state != -1 and min_action > -1
    #print('min ratio: ', min_ratio)
    if min_ratio < 1:
        return min_ratio ** TAU
    else:
        return min_ratio

# Loop over all states, all actions such that IS weight is maximized
def max_weight(pi_e, pi_b):
    actions = len(ACTIONS)
    max_state = -1
    max_action = -1
    max_ratio = -1
    for x in range(0, ROW):
        for y in range(0, COL):
            ind = coordinate_to_index(x, y)
            for a in range(0, actions): 
                ratio = pi_e[ind, a] / pi_b[ind, a]
                if(ratio > max_ratio):
                    max_ratio = ratio
                    max_state = (x, y)
                    max_action = a
    assert max_ratio > -1 and max_state != -1 and max_action > -1
    if max_ratio < 1:
        return max_ratio
    else:
        return max_ratio ** TAU

# pi_e is a new candidate policy
# pi_b is the original policy we have taken samples from
# D is a list of list of trajectories sampled from pi_b
def importance_sampling_estimates(pi_e, pi_b, D, directory):
    assert pi_e.shape == pi_b.shape
    is_weights = []
    rewards = []
    for trajectory_num in D:
        with open(directory+'/trajectory-'+str(trajectory_num)+'.p', 'rb') as f:
            trajectory = pickle.load(f)
            i = len(trajectory) - 1
            reward = trajectory[i][2] # the end of very last triplet is the reward
            #print('Trajectory len: ', len(trajectory))
            #assert reward > 0
            #assert len(trajectory) <= TAU
            prod = 1
            for i in range(0, len(trajectory)): # go through every triplet in a trajectory
                s, a, r = trajectory[i]
                ind = coordinate_to_index(s[0], s[1])
                assert ind > -1 and ind < ROW*COL
                prod *= (pi_e[ind, a] / pi_b[ind, a])
            is_weights.append(prod)
            rewards.append(reward)
    return np.asarray(is_weights), np.asarray(rewards)

def adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, directory, weighting, k, w, flag=False, c=-1):
    is_weights, rewards = importance_sampling_estimates(pi_e, pi_b, D_safety, directory)
    is_weights = np.append(is_weights, np.repeat(w, k)) # add w, k times
    rewards = np.append(rewards, np.repeat(1, k)) # add max reward of 1, k times
    if flag: # using Panacea
        is_weights[is_weights > c] = c # collapse IS weights to c
        estimates = is_weights * rewards
        b = c
    else: # not using Panacea
        estimates = is_weights * rewards
        b = w
    if weighting == 'WIS':
        m = len(D_safety) + k # total number of r.v.s including attacker's trajectories
        assert len(is_weights) == m
        total_is_weights = np.sum(is_weights)
        norm = (1/m) * total_is_weights
        assert norm > 0
        estimates = estimates * (1/norm)
        b = 1
    diff = CH(estimates, delta, b)
    return diff
