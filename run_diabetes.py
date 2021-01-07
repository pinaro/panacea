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
import os
import numpy as np
from diabetes import *
from constants import *

from gym.envs.registration import register
register(
    id='Pinar_SimGlucose-v0',
    entry_point='SimGlucose.simglucose.envs:Pinar_T1DSimEnv',
    kwargs={'patient_name':PATIENT}
)

env = gym.make('Pinar_SimGlucose-v0')

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
        env.reset() 
        observation, reward, done, info = env.step([cr, cf])
        if reward > maxi:
            maxi = reward
        if reward < mini:
            mini = reward
    return mini, maxi

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

# Mega distribution from which all policy parameters are chosen from
# Uniform distribution used
def pick_policy_distribution():
    cr_c = np.random.uniform(low=CR_LOW, high=CR_HIGH)
    cf_c = np.random.uniform(low=CF_LOW, high=CF_HIGH)
    return cr_c, cf_c

# Create datasets
def create_datasets(cr_c_b, cf_c_b):
    os.mkdir(PATH+'data/diabetes_b_policy/')
    for i in range(0, NUM_DATASETS):
        filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        with open(PATH+'data/diabetes_b_policy/'+filename+'.csv', 'w+') as f:
            f.write('CR\tCF\tSampledCR\tSampledCF\tReward\n')
            for j in range(0, NUM_POINTS_PER_DATASET):
                cr = np.random.triangular(left=CR_LOW, mode=cr_c_b, right=CR_HIGH, size=None)
                cf = np.random.triangular(left=CF_LOW, mode=cf_c_b, right=CF_HIGH, size=None)
                env.reset()
                observation, reward, done, info = env.step([cr, cf])
                f.write(str(cr_c_b)+'\t'+str(cf_c_b)+'\t'+str(cr)+'\t'+str(cf)+'\t'+str(reward)+'\n')
                f.flush()
                
def compute_J(cr_c, cf_c):
    filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    with open(PATH+'data/'+filename+'_J_'+str(cr_c)+'_'+str(cf_c)+'.csv', 'w+') as f:
        f.write('CR\tCF\tSampledCR\tSampledCF\tReward\n')
        for j in range(0, NUM_SAMPLES_FOR_J):
            cr = np.random.triangular(left=CR_LOW, mode=cr_c, right=CR_HIGH, size=None)
            cf = np.random.triangular(left=CF_LOW, mode=cf_c, right=CF_HIGH, size=None)
            env.reset()
            observation, reward, done, info = env.step([cr, cf])
            f.write(str(cr_c)+'\t'+str(cf_c)+'\t'+str(cr)+'\t'+str(cf)+'\t'+str(reward)+'\n')
            f.flush()
    total_reward = 0
    num_samples = 0
    with open(PATH+'data/'+filename+'_J_'+str(cr_c)+'_'+str(cf_c)+'.csv', 'r') as f:
        next(f) # skip header
        for line in f:
            [cr, cf, sampled_cr, sampled_cf, reward] = line.split('\t')
            r = float(reward[:-1]) # exclude '\n'
            total_reward += normalize_return(r)   
            num_samples += 1
    return total_reward / num_samples

def safety_tests(K, datasets, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation, highest_cr_ratio, highest_cf_ratio, attacker_reward):
    print('Computing results for safety tests...')
    with open(PATH+'results/without_panacea.csv', 'w+') as f:
        f.write('k\tEstimator\tResult\tProblem\n')
        for file in datasets: 
            is_weights, rewards = compute_IS_weights(file, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation)
            for k in K: # number of trajectories added inc: 
                adversarial_is_weights, adversarial_rewards = add_adversarial_trajectories(is_weights, rewards, (highest_cr_ratio * highest_cf_ratio), attacker_reward, k)
                ch_wis_weights = create_wis_weights(adversarial_is_weights)
                ch_is = CH(adversarial_is_weights * adversarial_rewards, DELTA, b=highest_cr_ratio*highest_cf_ratio)
                ch_wis = CH(ch_wis_weights * adversarial_rewards, DELTA, b=1)
                f.write(str(k)+'\tIS\t'+str(ch_is)+'\tDiabetes\n')
                f.write(str(k)+'\tWIS\t'+str(ch_wis)+'\tDiabetes\n')
                f.flush()  
    
def panacea(K, datasets, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation, highest_cr_ratio, highest_cf_ratio, lowest_cr_ratio, lowest_cf_ratio, attacker_reward):
    #I_max_diabetes = 1879.306869629937
    #I_min_diabetes = 0.09010563601580672
    print('Computing results for Panacea...')
    with open(PATH+'results/with_panacea.csv', 'w+') as f:
        f.write('k\tEstimator\tpanaceaAlpha\tClip\tResult\tProblem\n')
        for file in datasets: 
            for weighting in ['IS', 'WIS']:
                is_weights, rewards = compute_IS_weights(file, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation)
                for k in K[1:]: # number of trajectories added to D of size 1500
                    if weighting == 'IS':
                        Alpha = ALPHA_IS
                    else:
                        Alpha = ALPHA_WIS
                    for alpha in Alpha:
                        # Compute clipping weight based on weighting
                        if weighting == 'IS':
                            c = compute_c('CH', 'IS', alpha, k, 1500)
                        else:
                            c = compute_c('CH', 'WIS', alpha, k, 1500, lowest_cr_ratio*lowest_cf_ratio)
                        assert c > 0
        
                        adversarial_is_weights, adversarial_rewards = add_adversarial_trajectories(is_weights, rewards, (highest_cr_ratio * highest_cf_ratio), attacker_reward, k)
                        adversarial_is_weights[adversarial_is_weights > c] = c # clip IS weights only
                        if weighting == 'IS':
                            ch_is = CH(adversarial_is_weights * adversarial_rewards, DELTA, b=c)
                            f.write(str(k)+'\tIS\t'+str(alpha)+'\t'+str(c)+'\t'+str(ch_is)+'\tDiabetes\n')
                            f.flush()
                        else:
                            ch_wis_weights = create_wis_weights(adversarial_is_weights)
                            ch_wis = CH(ch_wis_weights * adversarial_rewards, DELTA, b=1)
                            f.write(str(k)+'\tWIS\t'+str(alpha)+'\t'+str(c)+'\t'+str(ch_wis)+'\tDiabetes\n')
                            f.flush()
    
def run_diabetes(arg): 
    if arg == '1': # pick random behavior and evaluation policy
        print('Behavior policy selected.')
        cr_c_b, cf_c_b = pick_policy_distribution() # behavior policy
        print('Creating datasets for behaviour policy...')
        create_datasets(cr_c_b, cf_c_b)
        print('Computing performance of behaviour policy...')
        J_behavior = compute_J(cr_c_b, cf_c_b)
        print('J(\pi_b): ', J_behavior)
        print('Finding a suitable evaluation policy...')
        flag = True
        while flag:
            cr_c_e, cf_c_e = pick_policy_distribution() # evalution policy 
            J_eval = compute_J(cr_c_e, cf_c_e)
            if J_eval < J_behavior:
                flag = False  
        print('Evaluation policy selected.')
        print('J(\pi_e): ', J_eval)
        cr_behavior, cf_behavior, cr_evaluation, cf_evaluation = create_rvs(cr_c_b, cf_c_b, cr_c_e, cf_c_e)
    else: # recreate results in paper
        print('Behavior policy selected.')
        print('Creating datasets for behaviour policy...')
        #create_datasets(CR_MEAN, CF_MEAN)
        print('Computing performance of behaviour policy...')
        #J_behavior = 0.21880416377956427
        J_behavior = compute_J(CR_MEAN, CF_MEAN)
        print('J(\pi_b): ', J_behavior)
        print('Computing performance of evaluation policy...')
        cr_c_e, cf_c_e = [22.15929135188824, 49.95430132738277] # evalution policy
        #J_eval = 0.14460582621276313
        J_eval = compute_J(cr_c_e, cf_c_e)
        print('J(\pi_e): ', J_eval)
        cr_behavior, cf_behavior, cr_evaluation, cf_evaluation = create_rvs(CR_MEAN, CF_MEAN, cr_c_e, cf_c_e)
        
    # Datasets of behavior policy
    filenames = os.listdir(PATH+'data/diabetes_b_policy/')
    datasets = [filename for filename in filenames if filename.endswith('.csv')]
    
    # Attacker adds k copies of this for CH+IS and CH+WIS
    highest_cr_ratio = attacker_strategy(CR_LOW, CR_HIGH, cr_behavior, cr_evaluation)
    highest_cf_ratio = attacker_strategy(CF_LOW, CF_HIGH, cf_behavior, cf_evaluation)
    lowest_cr_ratio = attacker_strategy(CR_LOW, CR_HIGH, cr_behavior, cr_evaluation, False)
    lowest_cf_ratio = attacker_strategy(CF_LOW, CF_HIGH, cf_behavior, cf_evaluation, False)
    attacker_reward = 1
    K = np.arange(0, 151, 1)
    
    safety_tests(K, datasets, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation, highest_cr_ratio, highest_cf_ratio, attacker_reward)
    panacea(K, datasets, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation, highest_cr_ratio, highest_cf_ratio, lowest_cr_ratio, lowest_cf_ratio, attacker_reward)
    return J_behavior, J_eval
    