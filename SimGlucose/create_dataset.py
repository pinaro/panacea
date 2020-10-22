#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:14:57 2020

@author: Pinar
@description: creating figs for NeurIPS
"""

import gym
import numpy as np
import random
import string

# Found patient-specific variables from /params/Quest.csv
PATIENT = 'adult#003'
CR_MEAN = 9
CF_MEAN = 17.9345522688
PATH = '/mnt/nfs/work1/brian/pinar/Safe-Secure-RL/dataset/'

MIN_RETURN = 1e+100
MAX_RETURN = -1e+100

NUM_SAMPLES = 10000

from gym.envs.registration import register
register(
    id='Pinar_SimGlucose-v0',
    entry_point='simglucose.envs:Pinar_T1DSimEnv',
    kwargs={'patient_name':PATIENT}
)

env = gym.make('Pinar_SimGlucose-v0')
CR_LOW, CF_LOW = env.action_space.low
CR_HIGH, CF_HIGH = env.action_space.high
print('cr low: ', CR_LOW)
print('cr high: ', CR_HIGH)
print('cf low: ', CF_LOW)
print('cf high: ', CF_HIGH)

def compute_J(cr_c, cf_c):
    total_reward = 0
    for i in range(0, NUM_SAMPLES):
        cr = np.random.triangular(left=CR_LOW, mode=cr_c, right=CR_HIGH, size=None)
        cf = np.random.triangular(left=CF_LOW, mode=cf_c, right=CF_HIGH, size=None)
        observation, reward, done, info = env.step([cr, cf])
        env.reset()
        total_reward += normalize_return(reward)
    return total_reward / NUM_SAMPLES

def compute_min_max_return():
    mini = MIN_RETURN
    maxi = MAX_RETURN
    for i in range(0, int(1e+7)):
        if (i % 1000) == 0:
            print('Iteration: ', i)
            print('Min return: ', mini)
            print('Max return: ', maxi)          
        cr = np.random.uniform(low=CR_LOW, high=CR_HIGH)
        cf = np.random.uniform(low=CF_LOW, high=CF_HIGH)
        env.reset() # does this need to happen?
        observation, reward, done, info = env.step([cr, cf])
        if reward > maxi:
            maxi = reward
        if reward < mini:
            mini = reward
    return mini, maxi

# Mega distribution from which all policy parameters are chosen from
# Uniform distribution used
def pick_policy_distribution():
    cr_c = np.random.uniform(low=CR_LOW, high=CR_HIGH)
    cf_c = np.random.uniform(low=CF_LOW, high=CF_HIGH)
    return cr_c, cf_c

def normalize_return(reward):
    assert reward >= MIN_RETURN and reward <= MAX_RETURN
    r = (reward - MIN_RETURN) / (MAX_RETURN + MIN_RETURN)
    #assert r >= 0 and r <= 1
    return r

def create_datasets():
    cr_c_b, cf_c_b = CR_MEAN, CF_MEAN # very good policy for this patient
    print('C of CR_b: ', cr_c_b)
    print('C of CF_b: ', cf_c_b)
    for i in range(0, 1000):
        filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        with open(PATH+filename+'.csv', 'w+') as f:
            f.write('CR\tCF\tSampledCR\tSampledCF\tReward\n')
            for j in range(0, NUM_SAMPLES):
                cr = np.random.triangular(left=CR_LOW, mode=cr_c_b, right=CR_HIGH, size=None)
                cf = np.random.triangular(left=CF_LOW, mode=cf_c_b, right=CF_HIGH, size=None)
                env.reset()
                observation, reward, done, info = env.step([cr, cf])
                f.write(str(cr_c_b)+'\t'+str(cf_c_b)+'\t'+str(cr)+'\t'+str(cf)+'\t'+str(reward)+'\n')
                f.flush()
    
def main():
    mini, maxi = compute_min_max_return()
    print('Min return: ', mini)
	print('Max return: ', maxi)
    create_datasets()
#    cr_c_e, cf_c_e = [12.653255221453112, 15.739143271001524]#pick_policy_distribution()
#    print('mu of CR_e: ', cr_mean_e)
#    print('mu of CF_e: ', cf_mean_e)
#    CR_e, CF_e = create_rvs_to_sample(cr_mean_e, cf_mean_e)
#    J_behavior = -29.61905515240097#compute_J(CR_b, CF_b)
#    J_eval = -31.473781681023755#compute_J(CR_e, CF_e)
#    print('J behavior: ', J_behavior)
#    print('J eval: ', J_eval)
#    plot_policy_dist(CR_b, CF_b, CR_e, CF_e)
    
if __name__=="__main__":
    main()