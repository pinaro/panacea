#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:19:31 2020

@author: pinar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:14:57 2020

@author: Pinar
@description: creating figs for NeurIPS
"""

from spi import *
from scipy import stats
import os
import numpy as np

# Found patient-specific variables from /params/Quest.csv
PATIENT = 'adult#003'
CR_MEAN = 9
CF_MEAN = 17.9345522688
PATH = '/Users/pinar/Desktop/NeurIPS_fig1/diabetes_b_policies/'#'/mnt/nfs/work1/brian/pinar/Safe-Secure-RL/'

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

def normalize_return(reward):
    #print('Reward: ', reward)
    assert reward >= MIN_RETURN and reward <= MAX_RETURN
    r = (reward - MIN_RETURN) / (MAX_RETURN + MIN_RETURN)
    r = -1 * r
    assert r >= 0 and r <= 1
    return r

def create_rvs(cr_c_e, cf_c_e):
    cr_behavior = stats.triang(c=(CR_MEAN-CR_LOW)/(CR_HIGH-CR_LOW), loc=CR_LOW, scale=CR_HIGH-CR_LOW)
    cf_behavior = stats.triang(c=(CF_MEAN-CF_LOW)/(CF_HIGH-CF_LOW), loc=CF_LOW, scale=CF_HIGH-CF_LOW)
    cr_evaluation = stats.triang(c=(cr_c_e-CR_LOW)/(CR_HIGH-CR_LOW), loc=CR_LOW, scale=CR_HIGH-CR_LOW)
    cf_evaluation = stats.triang(c=(cf_c_e-CF_LOW)/(CF_HIGH-CF_LOW), loc=CF_LOW, scale=CF_HIGH-CF_LOW)
    return cr_behavior, cf_behavior, cr_evaluation, cf_evaluation

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
    #print('Norm: ', norm)
    wis = copy * (1/norm)
    #print(sum(wis >= 0))
    assert sum(wis >= 0) == len(wis) # no weight is negative
    #print(wis)
    return wis

def compute_c(CI, weighting, alpha_panacea, k_panacea, n_panacea, i_min=-1):
    delta = DELTA
    if CI == 'CH' and weighting == 'IS':
        term1 = np.sqrt(np.log(1/delta) / (2*n_panacea))
        m = k_panacea + n_panacea
        term2 = np.sqrt(np.log(1/delta) / (2*m))
        term3 = k_panacea / m
        return alpha_panacea / (term1 - term2 + term3)
    else:
        term1 = np.sqrt(np.log(1/delta) / (2*n_panacea))
        m = k_panacea + n_panacea
        term2 = np.sqrt(np.log(1/delta) / (2*m))
        numerator = i_min * (alpha_panacea - term1 + term2)
        denom = k_panacea * ((1-alpha_panacea) + term1 - term2)
        return numerator / denom
    
def max_alpha_CH_IS(n, k, i_star):
    tmp1 = np.log(1/DELTA) / (2*n)
    m = n+k
    tmp2 = np.log(1/DELTA) / (2*m)
    term1 = np.sqrt(tmp1) - np.sqrt(tmp2) + (k/m)
    return i_star * term1

def max_alpha_CH_WIS(n, k, i_star, i_min):
    tmp = np.log(1/DELTA) / (2*n)
    term1 = np.sqrt(tmp)
    m = n+k
    tmp = np.log(1/DELTA) / (2*m)
    term2 = np.sqrt(tmp)
    tmp = (k * i_star)
    term3 = tmp / (i_min + tmp)
    return term1 - term2 + term3#+ 1
        
def main():    
    # Datasets of behavior policy
    filenames = os.listdir(PATH)
    datasets = [filename for filename in filenames if filename.endswith('.csv')]

    # Evaluation policy
    cr_c_e, cf_c_e = [22.15929135188824, 49.95430132738277]
    J_eval = 0.14460582621276313
    J_behavior = 0.21880416377956427
    
    # RVs for behavior and eval policy
    cr_behavior, cf_behavior, cr_evaluation, cf_evaluation = create_rvs(cr_c_e, cf_c_e)
    
    # Attacker adds k copies of this for CH+IS, CH+WIS, AM+IS
    highest_cr_ratio = attacker_strategy(CR_LOW, CR_HIGH, cr_behavior, cr_evaluation)
    print('Highest cr: ', highest_cr_ratio)
    highest_cf_ratio = attacker_strategy(CF_LOW, CF_HIGH, cf_behavior, cf_evaluation)
    print('Highest cf: ', highest_cf_ratio)
    # Attacker adds k copies of this for AM+WIS
    lowest_cr_ratio = attacker_strategy(CR_LOW, CR_HIGH, cr_behavior, cr_evaluation, False)
    print('Lowest cr: ', lowest_cr_ratio)
    lowest_cf_ratio = attacker_strategy(CF_LOW, CF_HIGH, cf_behavior, cf_evaluation, False)
    print('Lowest cf: ', lowest_cf_ratio)
    attacker_reward = 1
    
    K = np.arange(1, 151, 1)
    Alpha_WIS = [0.1, 0.5, 1]
    Alpha_IS = [0.085]#[0.1, 0.5, 1, 5]
    I_max_diabetes = 1879.306869629937
    I_min_diabetes = 0.09010563601580672
    
    with open('/Users/Pinar/Desktop/NeurIPS_fig1/diabetes_panacea_xtra.csv', 'w+') as f:
        f.write('k\tEstimator\tpanaceaAlpha\tClip\tResult\tProblem\n')
        for file in datasets: 
            print('Running', file, '...' )
            for weighting in ['IS']:
                is_weights, rewards = compute_IS_weights(file, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation)
                for k in K: # number of trajectories added to D of size 1500
                    if weighting == 'IS':
                        Alpha = Alpha_IS
                    else:
                        Alpha = Alpha_WIS
                    for alpha in Alpha:
                        # Compute clipping weight based on weighting
                        if weighting == 'IS':
                            c = compute_c('CH', 'IS', alpha, k, 1500)
                            #print('IS c: ', c)
                        else:
                            c = compute_c('CH', 'WIS', alpha, k, 1500, lowest_cr_ratio*lowest_cf_ratio)
                            #print('WIS c: ', c)
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

if __name__=="__main__":
    main()
                        