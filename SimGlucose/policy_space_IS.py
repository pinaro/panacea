#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:42:47 2020

@author: Pinar
@description: Beginning to code IS over policy space
"""

import gym
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from cma import fmin

# Found patient-specific variables from /params/Quest.csv
PATIENT = 'adolescent#002'
CR_MEAN = 8
CF_MEAN = 9.21276345633
SIGMA = 10
PATH = '/Users/Pinar/Desktop/'

NUM_SAMPLES = 100

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

# Truncated norm does not work here
#def pick_policy_distribution():
#    # CR
#    lower, upper = CR_LOW, CR_HIGH
#    mu, sigma = CR_MEAN, SIGMA
#    CR = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
#    # CF
#    lower, upper = CF_LOW, CF_HIGH
#    mu, sigma = CF_MEAN, SIGMA
#    CF = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
#    return CR.rvs(1)[0], CF.rvs(1)[0]
# Truncated norm does not work
#def create_rvs_to_sample(cr_mean, cf_mean):
#    lower, upper = CR_LOW, CR_HIGH
#    mu, sigma = cr_mean, SIGMA
#    CR = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
#    lower, upper = CF_LOW, CF_HIGH
#    mu, sigma = cf_mean, SIGMA
#    CF = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
#    return CR, CF

# Mega distribution from which all policy parameters are chosen from
# Uniform distribution used
def pick_policy_distribution():
    cr_c = np.random.uniform(low=CR_LOW, high=CR_HIGH)
    cf_c = np.random.uniform(low=CF_LOW, high=CF_HIGH)
    return cr_c, cf_c

def create_rvs_to_sample(cr_mean, cf_mean):
    lower, upper = CR_LOW, CR_HIGH
    mu, sigma = cr_mean, SIGMA
    CR = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    lower, upper = CF_LOW, CF_HIGH
    mu, sigma = cf_mean, SIGMA
    CF = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return CR, CF

def plot_policy_dist(CR_b, CF_b, CR_e, CF_e): # main():  
#    lower, upper = CR_LOW, CR_HIGH
#    mu, sigma = CR_MEAN, SIGMA
#    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
#    N = stats.norm(loc=mu, scale=sigma)
#    X.pdf(CR_MEAN)
#    
#    fig, ax = plt.subplots(2, sharex=True)
#    ax[0].hist(X.rvs(10000), normed=True)
#    ax[1].hist(N.rvs(10000), normed=True)
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    ax[0, 0].hist(CR_b.rvs(10000), normed=True)
    ax[1, 0].hist(CR_e.rvs(10000), normed=True)
    ax[0, 1].hist(CF_b.rvs(10000), normed=True)
    ax[1, 1].hist(CF_e.rvs(10000), normed=True)
    samples = np.linspace(start=CR_LOW, stop=CR_HIGH, num=NUM_SAMPLES, endpoint=True)
    vals = CR_e.pdf(samples) / CR_b.pdf(samples)
    index = np.argmax(vals)
    new_cr_mu = samples[index]
    print('new cr mu: ', new_cr_mu)
    print(vals)
    ax[0, 2].hist(vals, normed=True)
    samples = np.linspace(start=CF_LOW, stop=CF_HIGH, num=NUM_SAMPLES, endpoint=True)
    vals = CF_e.pdf(samples) / CF_b.pdf(samples)
    index = np.argmax(vals)
    new_cf_mu = samples[index]
    print('new cf mu: ', new_cf_mu)
    print(vals)
    ax[1, 2].hist(vals, normed=True)
    #plt.show()
    return new_cr_mu, new_cf_mu
    
def compute_J(CR, CF):
    total_reward = 0
    for i in range(0, NUM_SAMPLES):
        cr = CR.rvs(1)[0]
        cf = CF.rvs(1)[0]
        env.reset() # does this need to happen?
        observation, reward, done, info = env.step([cr, cf])
        total_reward += reward
    return total_reward / NUM_SAMPLES

# Objective function to hand in to any black-box search algorithm (CMA-ES)
# CMA-ES is a minimizer, so return negative reward
def objective_function(s, *args):
    behavior = args[0]
    evaluation = args[1]
    mu = s[0]
    return -1 * (behavior.pdf(mu) / evaluation.pdf(mu))
    
# restart_from_best = True, bipop = True, restarts = 8, 
# 'transformation': [lambda x: ((np.reshape(x, (16, 4))**2) / np.sum((np.reshape(x, (16, 4))**2), axis=1).reshape(16, 1)).flatten(), None]
# 'verb_log': 0, 'verbose': -9, 'verb_disp': 0 , restarts = 5
# Call CMA-ES to find good a good candidate
def attacker_strategy_CMAES(CR_b, CF_b, CR_e, CF_e):
    s = attacker_strategy(CR_LOW, CR_HIGH, CR_b, CR_e)
    es = fmin(objective_function, np.array([s]), 1.0, args=(CR_b, CR_e), options={'maxiter': 5, 'tolfun':1e-12})
    cr = es[0]
    s = attacker_strategy(CF_LOW, CF_HIGH, CF_b, CF_e)
    fmin(objective_function, np.array([s]), 1.0, args=(CF_b, CF_e), options={'maxiter': 5, 'tolfun':1e-12})
    cf = es[0]
    return cr, cf

# Behavior and evaluation are rvs, not parameters
def attacker_strategy(low, high, behavior, evaluation):
    samples = np.linspace(start=low, stop=high, num=NUM_SAMPLES, endpoint=True)
    IS_weights = evaluation.pdf(samples) / behavior.pdf(samples)
    #highest_weight = np.max(IS_weights)
    index = np.argmax(IS_weights)
    return index
    
def main():
    cr_mean_b, cf_mean_b = [4.178900281249689, 11.364997866327915]#pick_policy_distribution()
    CR_b, CF_b = create_rvs_to_sample(cr_mean_b, cf_mean_b)
#    print('mu of CR_b: ', cr_mean_b)
#    print('mu of CF_b: ', cf_mean_b)
#    with open(PATH+'diabates_data.csv', 'w+') as f:
#        f.write('CR\tCF\tSampledCR\tSampledCF\tReward\n')
#        for i in range(0, NUM_SAMPLES):
#            cr = CR_b.rvs(1)[0]
#            cf = CF_b.rvs(1)[0]
#            env.reset() # does this need to happen?
#            observation, reward, done, info = env.step([cr, cf])
#            f.write(str(cr_mean_b)+'\t'+str(cf_mean_b)+'\t'+str(cr)+'\t'+str(cf)+'\t'+str(reward)+'\n')
#            f.flush()
    cr_mean_e, cf_mean_e = [12.653255221453112, 15.739143271001524]#pick_policy_distribution()
    print('mu of CR_e: ', cr_mean_e)
    print('mu of CF_e: ', cf_mean_e)
    CR_e, CF_e = create_rvs_to_sample(cr_mean_e, cf_mean_e)
    J_behavior = -29.61905515240097#compute_J(CR_b, CF_b)
    J_eval = -31.473781681023755#compute_J(CR_e, CF_e)
    print('J behavior: ', J_behavior)
    print('J eval: ', J_eval)
    #cr_mu, cf_mu = attacker_strategy_CMAES(CR_b, CF_b, CR_e, CF_e)
    #print('CR mu to maximize ratio: ', cr_mu)
    #print('CF mu to mzaximize ratio: ', cf_mu)
    plot_policy_dist(CR_b, CF_b, CR_e, CF_e)

if __name__=="__main__":
    main()
    