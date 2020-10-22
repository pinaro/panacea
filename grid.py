# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:52:39 2017

@author: Pinar Ozisik
"""

from constants import *
import numpy as np
import pickle
import os
import string
import random

def coordinate_to_index(x, y): # needed for policy lookup
    return (x+y) + ((y*COL) - y)

def index_to_coordinate(index): # needed for adversarial tampering
    x = index // (ROW-1)
    y = index - ((ROW-1) * x)
    return (x, y)

def action_to_col_num(a):
    if a == 'UP':
        return 0
    elif a == 'DOWN':
        return 1
    elif a == 'RIGHT':
        return 2
    elif a == 'LEFT':
        return 3

# rows are the total number of actual_states
# cols are the total number of actions
def return_rand_policy():
    data = np.random.rand(ROW*COL, len(ACTIONS))
    vector = np.sum(data, axis=1)
    policy = data / vector.reshape((vector.shape[0], 1))
    return policy

def next_state(x, y, a):
    if a == 0:
        new_x = x-1 # up
        new_y = y
    elif a == 1:
        new_x = x+1 # down 
        new_y = y
    elif a == 2: # right
        new_x = x
        new_y = y+1 
    elif a == 3: # left
        new_x = x
        new_y = y-1 
    if check_valid_state(new_x, new_y):
        return (new_x, new_y) # move to new place
    else:
        return (x, y) # stay in current place

def check_valid_state(x, y):
    if x > -1 and x < ROW and y > -1 and y < COL:
        return True
    else:
        return False

def select_random_action(s, policy):
    rand_val = np.random.rand()
    ind = coordinate_to_index(s[0], s[1])
    probs = np.cumsum(policy[ind, :])
    for i in range(0, probs.shape[0]):
        if rand_val < probs[i]:
            return i
#        if i == 0: # first
#            if rand_val < probs[i]:
#                assert i > -1 and i < len(ACTIONS)
#                return i
#        elif i == probs.shape[0]-1: # last
#            print('i', i)
#            print('rand_val', rand_val)
#            print('probs[i]', probs[i])
#            assert rand_val <= probs[i]
##            try:
##                assert rand_val <= probs[i]
##            except AssertionError as error:
##                print('rand: ', rand_val)
##                print('i: ', i)
##                print('probs[i]: ', probs[i])
#            assert i > -1 and i < len(ACTIONS)
#            return i
#        else:
#            if rand_val < probs[i]:
#                assert i > -1 and i < len(ACTIONS)
#                return i

# Trajectory: [(s, a, r), (s, a, r), (s, a, r), ...]
# Each action is represented as an integer: 0 for UP, 1 for DOWN, 2 for RIGHT, 3 for LEFT
# flag = False means do not use tau; otherwise use tau
def sample_trajectories(policy, flag=False, num=EPISODES):
    new_directory = PATH+'trajectories-'+''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])
    os.mkdir(new_directory)
    #trajectories = []
    max_time = -1
    for episodes in range(0, num):
        s = START
        time = 0
        e = []
        while True:
            #print('state ', s)
            a = select_random_action(s, policy)
            #print('action ', a)
            s_prime = next_state(s[0], s[1], a)
            #print('s prime ', s_prime)
            if s_prime == FINISH:
                r = (GAMMA**time) * 1
                e.append((s, a, r))
                break
            #elif flag == True and time == TAU-1: # round is over
            #    r = 0
            #    e.append((s, a, r))
            #    break
            else:
                r = 0
                e.append((s, a, r))
                s = s_prime
                time += 1
        #print('Time: ', time)
        #assert time < TAU
        #print(e)
        pickle.dump(e, open(new_directory+'/trajectory-'+str(episodes)+'.p', 'wb'))
        if time > max_time:
            max_time = time
    #assert max_time > -1
    return new_directory+'/', max_time # returns directory where trajectories are kept
        #trajectories.append(e)
    #return trajectories

def compute_J(policy): 
    reward = 0
    for t in range(0, TRIALS):
        s = START
        time = 0
        e = []
        while True:
            a = select_random_action(s, policy)
            s_prime = next_state(s[0], s[1], a)
            if s_prime == FINISH:
                r = (GAMMA**time) * 1
                break
            #elif time == TAU-1: # round is over
            #    r = 0
            #    break
            else:
                r = 0
                s = s_prime
                time += 1
        #print('Time: ', time)
        #assert time < TAU
        reward += r
    return reward / TRIALS # returns avg reward
