#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:39:22 2020

@author: pinar
@description: Computes clipping weight of Panacea
"""

import numpy as np
from scipy import stats
from constants import *

# Chernoff-Hoeffding
def CH(estimates, delta, b):
    n = len(estimates)
    tmp = np.log(1/delta) / (2*n)
    tmp = b * np.sqrt(tmp)
    diff = np.mean(estimates) - tmp
    return diff

# New version Pinar did based on quadratic equation
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

def compute_IS_weights(file, cr_behavior, cr_evaluation, cf_behavior, cf_evaluation):
    #print('Filename: ', file)
    is_weights = []
    rewards = []
    with open(PATH+'data/diabetes_b_policy/'+file, 'r') as f:
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

def normalize_return(reward):
    #print('Reward: ', reward)
    assert reward >= MIN_RETURN and reward <= MAX_RETURN
    r = (reward - MIN_RETURN) / (MAX_RETURN + MIN_RETURN)
    r = -1 * r
    assert r >= 0 and r <= 1
    return r

def create_rvs(cr_c_b, cf_c_b, cr_c_e, cf_c_e):
    cr_behavior = stats.triang(c=(cr_c_b-CR_LOW)/(CR_HIGH-CR_LOW), loc=CR_LOW, scale=CR_HIGH-CR_LOW)
    cf_behavior = stats.triang(c=(cf_c_b-CF_LOW)/(CF_HIGH-CF_LOW), loc=CF_LOW, scale=CF_HIGH-CF_LOW)
    cr_evaluation = stats.triang(c=(cr_c_e-CR_LOW)/(CR_HIGH-CR_LOW), loc=CR_LOW, scale=CR_HIGH-CR_LOW)
    cf_evaluation = stats.triang(c=(cf_c_e-CF_LOW)/(CF_HIGH-CF_LOW), loc=CF_LOW, scale=CF_HIGH-CF_LOW)
    return cr_behavior, cf_behavior, cr_evaluation, cf_evaluation
