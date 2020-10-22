#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:14:57 2020

@author: Pinar
@description: creating figs for NeurIPS
"""

import gym
import random
import string
import scipy
import os
import numpy as np
#from spi import *

# Found patient-specific variables from /params/Quest.csv
PATIENT = 'adult#003'
CR_MEAN = 9
CF_MEAN = 17.9345522688
PATH = '/mnt/nfs/work1/brian/pinar/Safe-Secure-RL/dataset/'

MIN_RETURN = 1e+100
MAX_RETURN = -1e+100

NUM_SAMPLES = 10000
DELTA = 0.05

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

# Create datasets
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
                
def attacker_strategy(low, high, behavior, evaluation, flag=True):
    samples = np.linspace(start=low, stop=high, num=int(1e+8), endpoint=True)
    IS_weights = evaluation.pdf(samples) / behavior.pdf(samples)
    #index = np.argmax(IS_weights)
    if flag == True:
        highest_weight = np.max(IS_weights)
        return highest_weight
    else:
        lowest_weight = np.min(IS_weights)
        return lowest_weight
    
def create_rvs(cr_c_e, cf_c_e):
    cr_b_rv = scipy.stats.triang(c=CR_MEAN, loc=CR_LOW, scale=CR_HIGH-CR_LOW)
    cf_b_rv = scipy.stats.triang(c=CF_MEAN, loc=CF_LOW, scale=CF_HIGH-CF_LOW)
    cr_e_rv = scipy.stats.triang(c=cr_c_e, loc=CR_LOW, scale=CR_HIGH-CR_LOW)
    cf_e_rv = scipy.stats.triang(c=cf_c_e, loc=CF_LOW, scale=CF_HIGH-CF_LOW)
    return cr_b_rv, cf_b_rv, cr_e_rv, cf_e_rv
    
def compute_IS_weights(file, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation):
    is_weights = []
    rewards = []
    with open(PATH+file, 'r') as f:
        next(f) # skip header
        for line in f:
            [cr, cf, sampled_cr, sampled_cf, reward] = line.split('\t')
            cr_ratio = cr_evaluation.pdf(float(sampled_cr)) / cr_behavior.pdf(float(sampled_cr))
            cf_ratio = cf_evaluation.pdf(float(sampled_cf)) / cf_behavior.pdf(float(sampled_cf))
            r = float(reward[:-1]) # exclude '\n'
            normalized = normalize_return(r)
            is_weights.append(cr_ratio * cf_ratio)
            rewards.append(normalized)
    np.asarray(is_weights), np.asarray(rewards)
    
def add_adversarial_trajectories(is_weights, rewards, attacker_weight, attacker_reward, k):
    weights = np.append(is_weights, np.repeat(attacker_weight, k)) # add weight, k times
    r = np.append(rewards, np.repeat(attacker_reward, k))
    return weights, r

def create_wis_weights(is_weights):
    m = len(is_weights) # total number of r.v.s including attacker's trajectories
    norm = (1/m) * np.sum(is_weights)
    wis = is_weights * (1/norm)
    return wis

def create_eval_policies():
    for i in range(0, 10):
        filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        cr_c_e, cf_c_e = pick_policy_distribution()
        with open('/mnt/nfs/work1/brian/pinar/Safe-Secure-RL/eval_policies/'+filename+'.csv', 'w+') as f:
            f.write('CR\tCF\tSampledCR\tSampledCF\tReward\n')
            for j in range(0, 1000):
                cr = np.random.triangular(left=CR_LOW, mode=cr_c_e, right=CR_HIGH, size=None)
                cf = np.random.triangular(left=CF_LOW, mode=cf_c_e, right=CF_HIGH, size=None)
                env.reset()
                observation, reward, done, info = env.step([cr, cf])
                f.write(str(cr_c_e)+'\t'+str(cf_c_e)+'\t'+str(cr)+'\t'+str(cf)+'\t'+str(reward)+'\n')
                f.flush()
      
def main():
    create_eval_policies()
    
def run_experiment():
    #mini, maxi = compute_min_max_return()
    #print('Min return: ', mini)
    #print('Max return: ', maxi)
    #create_datasets()
    
    # Datasets of behavior policy
    filenames = os.listdir(PATH)
    datasets = [filename for filename in filenames if filename.endswith('.csv')]
    
    # Evaluation policy
    cr_c_e, cf_c_e = pick_policy_distribution()
    J_eval = compute_J(cr_c_e, cf_c_e)
    
    # RVs for behavior and eval policy
    cr_behavior, cf_behavior, cr_evaluation, cf_evaluation = create_rvs(cr_c_e, cf_c_e)
    
    # Attacker adds k copies of this for CH+IS, CH+WIS, AM+IS
    highest_cr_ratio = attacker_strategy(CR_LOW, CR_HIGH, cr_behavior, cr_evaluation)
    highest_cf_ratio = attacker_strategy(CF_LOW, CF_HIGH, cf_behavior, cf_evaluation)
    # Attacker adds k copies of this for AM+WIS
    lowest_cr_ratio = attacker_strategy(CR_LOW, CR_HIGH, CR_MEAN, cr_c_e, False)
    lowest_cf_ratio = attacker_strategy(CF_LOW, CF_HIGH, CF_MEAN, cf_c_e, False)
    attacker_reward = maxi
    
    # 0%, 0.1%, 1%, 5%, 10%, 25%
    K = [0, 10, 100, 500, 1000, 2500]
    
    with open(PATH+'diabetes_results.csv', 'w+') as f:
        f.write('k\tEstimator\tResult\n')
        for file in datasets:  
            is_weights, rewards = compute_IS_weights(file, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation)
            for k in K: # number of trajectories added inc: 
                adversarial_is_weights, adversarial_rewards = add_adversarial_trajectories(is_weights, rewards, (highest_cr_ratio * highest_cf_ratio), attacker_reward, k)
                ch_wis_weights = create_wis_weights(adversarial_is_weights)
                ch_is = CH(adversarial_is_weights * adversarial_rewards, DELTA)
                ch_wis = CH(ch_wis_weights * rewards, DELTA)
                am_is = AM(adversarial_is_weights * adversarial_rewards, DELTA)
                
                tmp_is_weights, adversarial_rewards = add_adversarial_trajectories(is_weights, rewards, (lowest_cr_ratio * lowest_cf_ratio), attacker_reward, k)
                am_wis_weights = create_wis_weights(tmp_is_weights)
                am_wis = AM(am_wis_weights * adversarial_rewards, DELTA)
                
                f.write(str(k)+'\tCH, IS\t'+str(ch_is)+'\n')
                f.write(str(k)+'\tCH, WIS\t'+str(ch_wis)+'\n')
                f.write(str(k)+'\tAM, IS\t'+str(am_is)+'\n')
                f.write(str(k)+'\tAM, WIS\t'+str(am_wis)+'\n')
                f.flush()        
    
if __name__=="__main__":
    main()
