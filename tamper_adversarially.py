#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:59:34 2019

@author: Pinar
@description: Tamper adversarially
"""

from constants import *
from grid import *
import numpy as np
import pickle

# Tamper with n trajectories -- change ONLY 1 value
def tamper_adversarially(D, num, directory):
    tmp = np.random.choice(D, num)
    for trajectory_num in tmp:
        filename = directory+'trajectory-'+str(trajectory_num)+'.p'
        trajectory = pickle.load(open(filename, 'rb'))
        i = len(trajectory) - 1
        tupl = (trajectory[i][0], trajectory[i][1], BIG_CONSTANT)
        trajectory[i] = tupl # change very last tuple that includes non-zero reward
        pickle.dump(trajectory, open(filename, 'wb'))

# Loop over all states, all actions such that IS weight is maximized
def adversarially_monotonic_trajectory(pi_e, pi_b):
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
    
# Loop over all states, all actions such that IS weight is minimized
def AM_WIS_trajectory(pi_e, pi_b):
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
    