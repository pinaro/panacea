# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:52:39 2017

@author: Pinar Ozisik
@description: Parts of code needed for safe policy improvement (SPI)
"""

from grid import *
from constants import *
from scipy.stats import t
from cma import fmin, CMAEvolutionStrategy
import numpy as np
import random
import pickle

# Convert result returned by CMA-ES into a policy with correct constraints
def redefine_policy(p):
    policy = np.reshape(p, (ROW*COL, len(ACTIONS)))
    data = np.absolute(policy)
    vector = np.sum(data, axis=1)
    policy = data / vector.reshape((vector.shape[0], 1))
    # print(policy)
    assert policy.shape == (ROW*COL, len(ACTIONS))
    return policy

# Split D into 2 separate trajectories
def split_D():
    safety = int(SPLIT * EPISODES)
    trajectories = list(range(0, EPISODES))
    random.shuffle(trajectories)
    D_safety = trajectories[:safety]
    D_candidate = trajectories[safety:]
    return D_candidate, D_safety
    #D_candidate = random.sample(range(0, EPISODES), EPISODES-safety)
    #return D_candidate

# pi_e is a new candidate policy
# pi_b is the original policy we have taken samples from
# D is a list of list of trajectories sampled from pi_b
def importance_sampling_estimates(pi_e, pi_b, D, directory):
    assert pi_e.shape == pi_b.shape
    is_estimates = []
    total_is_weights = 0
    for trajectory_num in D:
        with open(directory+'trajectory-'+str(trajectory_num)+'.p', 'rb') as f:
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
            is_estimates.append(prod * reward)
            total_is_weights += prod
    return np.asarray(is_estimates), total_is_weights

def importance_sampling_estimates_panacea(pi_e, pi_b, D, directory):
    assert pi_e.shape == pi_b.shape
    is_weights = []
    rewards = []
    for trajectory_num in D:
        with open(directory+'trajectory-'+str(trajectory_num)+'.p', 'rb') as f:
            trajectory = pickle.load(f)
            i = len(trajectory) - 1
            reward = trajectory[i][2] # the end of very last triplet is the reward
            #print('Trajectory len: ', len(trajectory))
            #assert reward > 0
            assert len(trajectory) <= TAU
            prod = 1
            for i in range(0, len(trajectory)): # go through every triplet in a trajectory
                s, a, r = trajectory[i]
                ind = coordinate_to_index(s[0], s[1])
                assert ind > -1 and ind < ROW*COL
                prod *= (pi_e[ind, a] / pi_b[ind, a])
            is_weights.append(prod)
            rewards.append(reward)
    return np.asarray(is_weights), np.asarray(rewards)

# pi_e is a new candidate policy
# pi_b is the original policy we have taken samples from
# D is a list of list of trajectories sampled from pi_b
def weighted_importance_sampling_estimates(pi_e, pi_b, D, directory):
    assert pi_e.shape == pi_b.shape
    is_estimates = []
    total_is_weights = 0
    for trajectory_num in D:
        with open(directory+'trajectory-'+str(trajectory_num)+'.p', 'rb') as f:
            trajectory = pickle.load(f)
            i = len(trajectory) - 1
            reward = trajectory[i][2] # the end of very last triplet is the reward
            assert reward > 0
            prod = 1
            for i in range(0, len(trajectory)): # go through every triplet in a trajectory
                s, a, r = trajectory[i]
                ind = coordinate_to_index(s[0], s[1])
                assert ind > -1 and ind < ROW*COL
                prod *= (pi_e[ind, a] / pi_b[ind, a])
            is_estimates.append(prod * reward)
            total_is_weights += prod
    norm = (1/len(D)) * total_is_weights
    norm = 1/norm
    wis_estimates = []
    for w in is_estimates:
        wis_estimates.append(w * norm)
    return np.asarray(wis_estimates)

# Application of different CIs
def CH(estimates, delta, b):
    n = len(estimates)
    tmp = np.log(1/delta) / (2*n)
    tmp = b * np.sqrt(tmp)
    diff = np.mean(estimates) - tmp
    return diff
    
def AM(estimates, delta):
    n = len(estimates)
    tmp = np.log(2/delta) / (2*n)
    tmp = np.sqrt(tmp)
    estimates.sort()
    indices = np.arange(0, len(estimates), dtype=float)
    indices = (indices / len(estimates)) + tmp
    indices = np.minimum(1, indices)
    estimates = np.insert(estimates, 0, 0) # add zero at the beginning
    #print(len(np.diff(estimates)))
    #print(len(indices))
    assert len(np.diff(estimates)) == len(indices)
    total = np.sum(np.diff(estimates) * indices)
    #print('total', total)
    #print('last', estimates[-1])
    return (estimates[-1] - total)

def validate_AM_math(estimates, total_is_weights):
    n = len(estimates)
    increments = np.arange(2, 250, 1)
    for k in increments:
        tmp = np.log(2/DELTA) / (2*(n+k))
        tmp = np.sqrt(tmp)
        indices_former = np.arange(1, n+k, dtype=float)
        indices_former = (indices_former / (n+k)) + tmp
        indices_former = np.minimum(1, indices_former)
        indices_latter = np.arange(0, n+k-1, dtype=float)
        indices_latter = (indices_latter / (n+k)) + tmp
        indices_latter = np.minimum(1, indices_latter)
        g = indices_former - indices_latter
        #print('nm: ', g[-(k-1):])
        g_k_sum = np.sum(g[-(k-1):]) # last k vals
        numerator = np.sum(estimates * g[:n]) # first n items
        #print('k: ', k)
        #print('LHS: ', g_k_sum / k)
        #print('RHS: ', (numerator / total_is_weights))
        #print('Check: ', (g_k_sum / k) > (numerator / total_is_weights))
        if (g_k_sum / k) > (numerator / total_is_weights):
            #print('LHS: ', g_k_sum / k)
            #print('RHS: ', (numerator / total_is_weights))
            print('k found: ', k)
            break
    
def students_t_test(estimates, delta, D_safety):
    n = len(estimates)
    critical_val = t.ppf(1-delta, n-1)
    tmp = (np.std(estimates, ddof=1) / np.sqrt(n)) * critical_val # Bessel corrected
    diff = np.mean(estimates) - tmp
    return diff

# https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
# Returns True or False
def test_candidate_policy(pi_e, pi_b, D_candidate, J_pi_b, delta, is_estimates):
    n = int(SPLIT * EPISODES) # degrees of freedom
    critical_val = t.ppf(1-delta, n-1)
    tmp = 2 * (np.std(is_estimates, ddof=1) / np.sqrt(n)) * critical_val # Bessel corrected
    diff = np.mean(is_estimates) - tmp
#    if diff >= J_pi_b:
#        print('SOLUTION FOUND.')
#    else:
#        print('NO SOLUTION')
    return diff, diff >= J_pi_b
    # diff = np.mean(is_estimates) - tmp
    # is_estimates = importance_sampling_estimates(pi_e, pi_b, D_candidate)
    
# Objective function to hand in to any black-box search algorithm (CMA-ES)
# CMA-ES is a minimizer, so return negative reward
def objective_function(p, *args):
    pi_b = args[0]
    D_candidate = args[1]
    J_pi_b = args[2]
    delta = args[3]
    directory = args[4]
    pi_e = redefine_policy(p)
    is_estimates = importance_sampling_estimates(pi_e, pi_b, D_candidate, directory)
    diff, flag = test_candidate_policy(pi_e, pi_b, D_candidate, J_pi_b, delta, is_estimates)
    if flag:
        return -1 * np.mean(is_estimates)
    else:
        return (-1 * diff) + DISCOUNT_FACTOR
    
# restart_from_best = True, bipop = True, restarts = 8, 
# 'transformation': [lambda x: ((np.reshape(x, (16, 4))**2) / np.sum((np.reshape(x, (16, 4))**2), axis=1).reshape(16, 1)).flatten(), None]
# 'verb_log': 0, 'verbose': -9, 'verb_disp': 0 , restarts = 5
# Call CMA-ES to find good a good candidate
def return_CMAES_candidate(pi_b, D_candidate, J_pi_b, delta, directory):
    b = pi_b.flatten() #ROW*COL*len(ACTIONS)*[0]
    es = fmin(objective_function, b, 1.0, args=(pi_b, D_candidate, J_pi_b, delta, directory), options={'maxiter': 5, 'tolfun':1e-12})
    return es[0]

def adversarial_safety_test_panacea(pi_e, pi_b, D_safety, J_pi_b, delta, directory, weighting, CI, k, w, c):
    is_weights, rewards = importance_sampling_estimates_panacea(pi_e, pi_b, D_safety, directory)
    #print('CI: ', CI)
    #print('weighting: ', weighting)
    #print('TAU: ', TAU)
    #print('w: ', w)
    is_weights = np.append(is_weights, np.repeat(w, k)) # add w, k times
    rewards = np.append(rewards, np.repeat(1, k)) # add max reward of 1, k times
    is_weights[is_weights > c] = c # collapse IS weights to c
    estimates = is_weights * rewards
    #print('Len of estimates: ', len(estimates))
    b = c
    if weighting == 'WIS':
        m = len(D_safety) + k # total number of r.v.s including attacker's trajectories
        assert len(is_weights) == m
        total_is_weights = np.sum(is_weights)
        norm = (1/m) * total_is_weights
        assert norm > 0
        estimates = estimates * (1/norm)
        b = 1
    if CI == 'STT':
        diff = students_t_test(estimates, delta)
    elif CI == 'CH':
        diff = CH(estimates, delta, b)
    else:
        diff = AM(estimates, delta)
    #print('Lower bound performance: '+str(diff))
    #print('J_pi_b: '+str(J_pi_b))
    #if diff >= J_pi_b:
    #    print("Better policy found")
        #return pi_e, True
    #else:
    #    print("Keep current policy")
        #return pi_b, False
    return diff

def adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, directory, weighting, CI, k, w):
    estimates, total_is_weights = importance_sampling_estimates(pi_e, pi_b, D_safety, directory)
    #validate_AM_math(estimates, total_is_weights)
    #print('IS weight total: ', total_is_weights)
    assert total_is_weights > 0 
    #print('CI: ', CI)
    #print('weighting: ', weighting)
    #print('TAU: ', TAU)
    #print('w: ', w)
    estimates = np.append(estimates, np.repeat(w, k)) # add w, k times
    #print('Len of estimates: ', len(estimates))
    b = w
    if weighting == 'WIS':
        m = len(D_safety) + k # total number of r.v.s including attacker's trajectories
        norm = (1/m) * (total_is_weights + (k*w))
        assert norm > 0
        estimates = estimates * (1/norm)
        b = 1
    if CI == 'STT':
        diff = students_t_test(estimates, delta)
    elif CI == 'CH':
        diff = CH(estimates, delta, b)
    else:
        diff = AM(estimates, delta)
    #print('Lower bound performance: '+str(diff))
    #print('J_pi_b: '+str(J_pi_b))
    #if diff >= J_pi_b:
    #    print("Better policy found")
        #return pi_e, True
    #else:
    #    print("Keep current policy")
        #return pi_b, False
    return diff

def safety_test_tmp(pi_e, pi_b, D_safety, delta, directory, weighting, CI):
    if weighting == 'IS':
        estimates = importance_sampling_estimates(pi_e, pi_b, D_safety, directory)
    else: # WIS
        estimates = weighted_importance_sampling_estimates(pi_e, pi_b, D_safety, directory)
    if CI == 'STT':
        diff = students_t_test(estimates, delta, D_safety)
    elif CI == 'CH':
        diff = CH(estimates, delta, D_safety)
    else:
        diff = AM(estimates, delta, D_safety)
    print('Diff: '+str(diff))
    return diff
#    if diff >= J_pi_b:
#        print("Better policy found")
#        return pi_e, True
#    else:
#        print("Keep current policy")
#        return pi_b, False

def safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, directory, weighting, CI, k, J_hat):
    if weighting == 'IS':
        estimates = importance_sampling_estimates(pi_e, pi_b, D_safety, directory)
    else: # WIS
        estimates = weighted_importance_sampling_estimates(pi_e, pi_b, D_safety, directory)
    if CI == 'STT':
        diff = students_t_test(estimates, delta, D_safety)
    elif CI == 'CH':
        diff = CH(estimates, delta, D_safety)
    else:
        diff = AM(estimates, delta, D_safety)
    print('Diff: '+str(diff))
    if diff >= J_pi_b:
        print("Better policy found")
        return pi_e, True
    else:
        print("Keep current policy")
        return pi_b, False
        
# Full SPI algorithm
def safe_policy_improvement(delta, pi_b, J_pi_b):
    directory = sample_trajectories(pi_b)
    D_candidate, D_safety = split_D()
    pi_e = return_CMAES_candidate(pi_b, D_candidate, J_pi_b, delta, directory)
    if pi_e is not None and sum(np.isnan(pi_e)) == 0: # lazy python evaluation
        pi_e = redefine_policy(pi_e)
        return safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, directory)
    else:
        print('Error happened. Retry.')
        return safe_policy_improvement(delta, pi_b, J_pi_b)
