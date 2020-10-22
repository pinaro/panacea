#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:14:57 2020

@author: Pinar
@description: creating figs for NeurIPS
"""

from spi import *
from scipy import stats
#import gym
import random
import string
import os
import numpy as np
import matplotlib.pyplot as plt

# Found patient-specific variables from /params/Quest.csv
PATIENT = 'adult#003'
CR_MEAN = 9
CF_MEAN = 17.9345522688
PATH = '/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_b_policies/'#'/mnt/nfs/work1/brian/pinar/Safe-Secure-RL/'

#MIN_RETURN = 1e+100
#MAX_RETURN = -1e+100
MIN_RETURN = -45
MAX_RETURN = -20

NUM_SAMPLES = 10000
DELTA = 0.05

#from gym.envs.registration import register
#register(
#    id='Pinar_SimGlucose-v0',
#    entry_point='simglucose.envs:Pinar_T1DSimEnv',
#    kwargs={'patient_name':PATIENT}
#)
#
#env = gym.make('Pinar_SimGlucose-v0')
CR_LOW, CF_LOW = 3, 5
CR_HIGH, CF_HIGH = 30, 50
#print('cr low: ', CR_LOW)
#print('cr high: ', CR_HIGH)
#print('cf low: ', CF_LOW)
#print('cf high: ', CF_HIGH)

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
    #print('Reward: ', reward)
    assert reward >= MIN_RETURN and reward <= MAX_RETURN
    r = (reward - MIN_RETURN) / (MAX_RETURN + MIN_RETURN)
    r = -1 * r
    assert r >= 0 and r <= 1
    return r

# Create datasets
def create_datasets():
    cr_c_b, cf_c_b = CR_MEAN, CF_MEAN # very good policy for this patient
    print('C of CR_b: ', cr_c_b)
    print('C of CF_b: ', cf_c_b)
    for i in range(0, 1000):
        filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        with open(PATH+'dataset/'+filename+'.csv', 'w+') as f:
            f.write('CR\tCF\tSampledCR\tSampledCF\tReward\n')
            for j in range(0, NUM_SAMPLES):
                cr = np.random.triangular(left=CR_LOW, mode=cr_c_b, right=CR_HIGH, size=None)
                cf = np.random.triangular(left=CF_LOW, mode=cf_c_b, right=CF_HIGH, size=None)
                env.reset()
                observation, reward, done, info = env.step([cr, cf])
                f.write(str(cr_c_b)+'\t'+str(cf_c_b)+'\t'+str(cr)+'\t'+str(cf)+'\t'+str(reward)+'\n')
                f.flush()
    
def create_rvs(cr_c_e, cf_c_e):
    cr_behavior = stats.triang(c=(CR_MEAN-CR_LOW)/(CR_HIGH-CR_LOW), loc=CR_LOW, scale=CR_HIGH-CR_LOW)
    cf_behavior = stats.triang(c=(CF_MEAN-CF_LOW)/(CF_HIGH-CF_LOW), loc=CF_LOW, scale=CF_HIGH-CF_LOW)
    cr_evaluation = stats.triang(c=(cr_c_e-CR_LOW)/(CR_HIGH-CR_LOW), loc=CR_LOW, scale=CR_HIGH-CR_LOW)
    cf_evaluation = stats.triang(c=(cf_c_e-CF_LOW)/(CF_HIGH-CF_LOW), loc=CF_LOW, scale=CF_HIGH-CF_LOW)
    return cr_behavior, cf_behavior, cr_evaluation, cf_evaluation

def create_eval_policies():
    for i in range(0, 10):
        filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        cr_c_e, cf_c_e = pick_policy_distribution()
        with open(PATH+'eval_policies/'+filename+'.csv', 'w+') as f:
            f.write('CR\tCF\tSampledCR\tSampledCF\tReward\n')
            for j in range(0, 1000):
                cr = np.random.triangular(left=CR_LOW, mode=cr_c_e, right=CR_HIGH, size=None)
                cf = np.random.triangular(left=CF_LOW, mode=cf_c_e, right=CF_HIGH, size=None)
                env.reset()
                observation, reward, done, info = env.step([cr, cf])
                f.write(str(cr_c_e)+'\t'+str(cf_c_e)+'\t'+str(cr)+'\t'+str(cf)+'\t'+str(reward)+'\n')
                f.flush()
                
def main():#create_eval_policies():
    filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    cr_c_e, cf_c_e = 22.15929135188824, 49.95430132738277
    with open(PATH+filename+'.csv', 'w+') as f:
        f.write('CR\tCF\tSampledCR\tSampledCF\tReward\n')
        for j in range(0, 1000):
            cr = np.random.triangular(left=CR_LOW, mode=cr_c_e, right=CR_HIGH, size=None)
            cf = np.random.triangular(left=CF_LOW, mode=cf_c_e, right=CF_HIGH, size=None)
            env.reset()
            observation, reward, done, info = env.step([cr, cf])
            f.write(str(cr_c_e)+'\t'+str(cf_c_e)+'\t'+str(cr)+'\t'+str(cf)+'\t'+str(reward)+'\n')
            f.flush()
                
def compute_J(path):
    total_reward = 0
    num_samples = 0
    with open(path, 'r') as f:
        next(f) # skip header
        for line in f:
            [cr, cf, sampled_cr, sampled_cf, reward] = line.split('\t')
            r = float(reward[:-1]) # exclude '\n'
            total_reward += normalize_return(r)   
            num_samples += 1
    print('CR: ', cr)
    print('CF: ', cf)
    print('J: ', total_reward / num_samples)
    print('---------------------------------')
    return total_reward / num_samples

def compute_J_behavior():
    total_reward = 0
    num_samples = 0
    filenames = os.listdir('/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_b_policies_xtra/')
    datasets = [filename for filename in filenames if filename.endswith('.csv')]
    for file in datasets:
        with open('/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_b_policies_xtra/'+file, 'r') as f:
            next(f) # skip header
            for line in f:
                [cr, cf, sampled_cr, sampled_cf, reward] = line.split('\t')
                r = float(reward[:-1]) # exclude '\n'
                total_reward += normalize_return(r)   
                num_samples += 1
                if num_samples == 10000:
                    break        
        print('CR: ', cr)
        print('CF: ', cf)
        print('J: ', total_reward / num_samples)
        print('---------------------------------')
        return total_reward / num_samples

def compute_min_max_return_from_data(path, mini, maxi):
    with open(path, 'r') as f:
        next(f) # skip header
        for line in f:
            [cr, cf, sampled_cr, sampled_cf, reward] = line.split('\t')
            r = float(reward[:-1]) # exclude '\n'
            if r > maxi:
                maxi = r
            if r < mini:
                mini = r
    return mini, maxi

def investigate_params():
    #mini, maxi = compute_min_max_return_from_data('/Users/Pinar/Desktop/behavior_policy.csv', MIN_RETURN, MAX_RETURN)
    #print('---------------------------------')
    #compute_J('/Users/Pinar/Desktop/behavior_policy.csv')
    print('---------------------------------')
    filenames = os.listdir('/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_eval_policies/')
    datasets = [filename for filename in filenames if filename.endswith('.csv')]
    mini_J = 100000
    mini = 100000000000
    maxi = -10
    for file in datasets: 
        m = compute_J('/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_eval_policies/'+file)
        if m < mini_J:
            mini_J = m
        mini, maxi = compute_min_max_return_from_data('/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_eval_policies/'+file, mini, maxi)
    print('Smallest J: ', mini_J)
    print('Smallest reward: ', mini)
    print('Largest reward: ', maxi)
    
def break_into_smaller_files(path):
    line_count = -1
    f1 = open('/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_b_policies/tmp.csv', 'w+')
    with open(path, 'r') as f:
        header = f.readline()
        line_count += 1
        for line in f:
            if (line_count % 1501) == 0:
                if not f1.closed:
                    f1.close()
                filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
                f1 = open('/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_b_policies/'+filename+'.csv', 'w+')
                f1.write(header)
            else:
                f1.write(line)
            line_count += 1
        
# Make sure there is only one header in the file before running script!
def create_small_datasets():
    break_into_smaller_files('/Users/Pinar/Desktop/NeurIPS_fig1/behavior_policy.csv')
    
def attacker_strategy(low, high, behavior, evaluation, flag=True):
    samples = np.linspace(start=low, stop=high, num=int(1e+7), endpoint=True)
    IS_weights = evaluation.pdf(samples) / behavior.pdf(samples)
    #index = np.argmax(IS_weights)
    if flag == True:
        highest_weight = np.nanmax(IS_weights)
        return highest_weight
    else:
        lowest_weight = np.nanmin(IS_weights)
        return lowest_weight
    
# Plot distributions
def plot_policy_dist():
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax[0, 1].hist(np.random.triangular(left=CR_LOW, mode=CR_MEAN, right=CR_HIGH, size=100000), bins=200, normed=True)
    ax[1, 1].hist(np.random.triangular(left=CF_LOW, mode=CF_MEAN, right=CF_HIGH, size=100000), bins=200, normed=True)
    ax[0, 0].hist(stats.triang.rvs(c=(CR_MEAN-CR_LOW)/(CR_HIGH-CR_LOW), loc=CR_LOW, scale=CR_HIGH-CR_LOW, size=100000), bins=2000, normed=True)
    ax[1, 0].hist(stats.triang.rvs(c=(CF_MEAN-CF_LOW)/(CF_HIGH-CF_LOW), loc=CF_LOW, scale=CF_HIGH-CF_LOW, size=100000), bins=2000, normed=True)
    plt.show()
    
def compute_IS_weights(file, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation):
    #print('Filename: ', file)
    is_weights = []
    rewards = []
    with open(PATH+file, 'r') as f:
        next(f) # skip header
        for line in f:
            #print(line)
            [cr, cf, sampled_cr, sampled_cf, reward] = line.split('\t')
            cr_ratio = cr_evaluation.pdf(float(sampled_cr)) / cr_behavior.pdf(float(sampled_cr))
            cf_ratio = cf_evaluation.pdf(float(sampled_cf)) / cf_behavior.pdf(float(sampled_cf))
            if reward[:-1] == '\n':
                r = float(reward[:-1]) # exclude '\n'
            else:
                r = float(reward)
            normalized = normalize_return(r)
            is_weights.append(cr_ratio * cf_ratio)
            rewards.append(normalized)
    assert sum(np.asarray(is_weights) >= 0) == len(np.asarray(is_weights)) # no weight is negative
    assert sum(np.asarray(rewards) >= 0) == len(np.asarray(rewards)) # no reward is negative
    return np.asarray(is_weights), np.asarray(rewards)

def add_adversarial_trajectories(is_weights, rewards, attacker_weight, attacker_reward, k):
    w = np.copy(is_weights)
    w = np.append(w, np.repeat(attacker_weight, k)) # add weight, k times
    r = np.copy(rewards)
    r = np.append(r, np.repeat(attacker_reward, k))
    return w, r

def create_wis_weights(is_weights):
    copy = np.copy(is_weights)
    m = len(is_weights) # total number including attacker's trajectories
    norm = (1/m) * np.sum(is_weights)
    wis = copy * (1/norm)
    assert sum(wis >= 0) == len(wis) # no weight is negative
    #print(wis)
    return wis
    
def main_tmp():
    #mini, maxi = compute_min_max_return()
    #print('Min return: ', mini)
    #print('Max return: ', maxi)
    #create_datasets()
    #create_eval_policies()
    
    # Datasets of behavior policy
    filenames = os.listdir(PATH)
    datasets = [filename for filename in filenames if filename.endswith('.csv')]
    
    # Evaluation policy
    cr_c_e, cf_c_e = [22.15929135188824, 49.95430132738277]
    J_eval = 0.14460582621276313
    J_behavior = 0.21880416377956427
    
    # RVs for behavior and eval policy
    cr_behavior, cf_behavior, cr_evaluation, cf_evaluation = create_rvs(cr_c_e, cf_c_e)
    #print('Prob CR: ', cr_behavior.pdf(CR_MEAN))
    #print('Prob CF: ', cf_behavior.pdf(CF_MEAN))
    
    # Attacker adds k copies of this for CH+IS, CH+WIS, AM+IS
    highest_cr_ratio = attacker_strategy(CR_LOW, CR_HIGH, cr_behavior, cr_evaluation)
    #print('Highest cr: ', highest_cr_ratio)
    highest_cf_ratio = attacker_strategy(CF_LOW, CF_HIGH, cf_behavior, cf_evaluation)
    #print('Highest cf: ', highest_cf_ratio)
    # Attacker adds k copies of this for AM+WIS
    lowest_cr_ratio = attacker_strategy(CR_LOW, CR_HIGH, cr_behavior, cr_evaluation, False)
    #print('Lowest cr: ', lowest_cr_ratio)
    lowest_cf_ratio = attacker_strategy(CF_LOW, CF_HIGH, cf_behavior, cf_evaluation, False)
    #print('Lowest cf: ', lowest_cf_ratio)
    attacker_reward = 1
    
    # 0%, 0.1%, 1%, 5%, 10%, 25%
    #K = [0, 1, 10, 50, 100, 250, 500]
    #K = [0, 10, 100, 500, 1000, 2500]
    K = np.arange(0, 151, 1)
    
    with open('/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_results_xtra.csv', 'w+') as f:
        f.write('k\tEstimator\tResult\tProblem\n')
        for file in datasets: 
            is_weights, rewards = compute_IS_weights(file, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation)
            for k in K: # number of trajectories added inc: 
                adversarial_is_weights, adversarial_rewards = add_adversarial_trajectories(is_weights, rewards, (highest_cr_ratio * highest_cf_ratio), attacker_reward, k)
                #print('A is weights len: ', len(adversarial_is_weights))
                #print('A rewards len: ',len(adversarial_rewards))
                ch_wis_weights = create_wis_weights(adversarial_is_weights)
                #print('A ch wis weights len: ', len(ch_wis_weights))
                ch_is = CH(adversarial_is_weights * adversarial_rewards, DELTA, b=highest_cr_ratio*highest_cf_ratio)
                #am_is = AM(adversarial_is_weights * adversarial_rewards, DELTA)
                ch_wis = CH(ch_wis_weights * adversarial_rewards, DELTA, b=1)
                
                # Do something a bit more complicated for AM+WIS (compute both scenarios and pick better one)
                #tmp_is_weights, adversarial_rewards = add_adversarial_trajectories(is_weights, rewards, (lowest_cr_ratio * lowest_cf_ratio), attacker_reward, k)
                #am_wis_weights = create_wis_weights(tmp_is_weights)
                #low_ratio_am_wis = AM(am_wis_weights * adversarial_rewards, DELTA)
                #high_ratio_am_wis = AM(ch_wis_weights * adversarial_rewards, DELTA)
                #tmp_is_weights, adversarial_rewards = add_adversarial_trajectories(is_weights, rewards, (lowest_cr_ratio * lowest_cf_ratio), 0.99, k)
                #am_wis_weights = create_wis_weights(tmp_is_weights)
                #third_option = AM(am_wis_weights * adversarial_rewards, DELTA)

                #if low_ratio_am_wis > high_ratio_am_wis:
                #    am_wis = low_ratio_am_wis
                #else:
                #    am_wis = high_ratio_am_wis 
                
                f.write(str(k)+'\tIS\t'+str(ch_is)+'\tDiabetes\n')
                f.write(str(k)+'\tWIS\t'+str(ch_wis)+'\tDiabetes\n')
                #f.write(str(k)+'\tAM, IS\t'+str(am_is)+'\n')
                #f.write(str(k)+'\tAM, WIS\t'+str(am_wis)+'\n')
                f.flush()        
    
if __name__=="__main__":
    main()

#print('-----------------------')
#print('K: ', k)
#if low_ratio_am_wis == max([low_ratio_am_wis, high_ratio_am_wis, third_option]):
#    am_wis = low_ratio_am_wis
#    print('Lowest selected')
#elif high_ratio_am_wis == max([low_ratio_am_wis, high_ratio_am_wis, third_option]):
#    am_wis = high_ratio_am_wis
#    print('Highest selected')
#else:
#    am_wis = third_option
#    print('Third option')
#print('-----------------------')