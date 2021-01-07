#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:38:53 2020

@author: pinar
@description: Panacea: does it work?
"""

import os
import numpy as np
from spi_grid import *
from constants import *
from grid import *

def create_pi_b(): # Use good policy as tester
    pi_e = np.full((ROW*COL, len(ACTIONS)), 0.25)
    
    pi_e[0, 2] = 0.85 # right at state 0
    pi_e[0, 0] = 0.05
    pi_e[0, 1] = 0.05
    pi_e[0, 3] = 0.05
 
    pi_e[3, 2] = 0.05 # down at state 3
    pi_e[3, 0] = 0.05
    pi_e[3, 1] = 0.85
    pi_e[3, 3] = 0.05
    
    pi_e[4, 1] = 0.05 # right at state 4
    pi_e[4, 2] = 0.85
    pi_e[4, 3] = 0.05
    pi_e[4, 0] = 0.05
    
    pi_e[7, 2] = 0.05 # down at state 7
    pi_e[7, 0] = 0.05
    pi_e[7, 1] = 0.85
    pi_e[7, 3] = 0.05
    
    return pi_e

def create_pi_e():
    pi_e = np.full((ROW*COL, len(ACTIONS)), 0.25)
    
    pi_e[0, 2] = 0.85 # right at state 0
    pi_e[0, 0] = 0.05
    pi_e[0, 1] = 0.05
    pi_e[0, 3] = 0.05
 
    pi_e[3, 2] = 0.05 # down at state 3
    pi_e[3, 0] = 0.05
    pi_e[3, 1] = 0.85
    pi_e[3, 3] = 0.05
    
    pi_e[4, 1] = 0.05 # right at state 4
    pi_e[4, 2] = 0.85
    pi_e[4, 3] = 0.05
    pi_e[4, 0] = 0.05
    
    pi_e[7, 2] = 0.2 # down at state 7
    pi_e[7, 0] = 0.2
    pi_e[7, 1] = 0.5
    pi_e[7, 3] = 0.1
    
    return pi_e

def create_datasets(pi):
    os.mkdir(PATH+'data/grid_b_policy/')
    for i in range(0, NUM_DATASETS):
        directory, max_time = sample_trajectories(pi, NUM_POINTS_PER_DATASET)

def compute_tau():
    filenames = os.listdir(PATH)
    datasets = [filename for filename in filenames if filename.startswith('trajectories')]
    maximum_length = 0
    for directory in datasets:
        print('Running ', directory)
        for trajectory_num in range(0, 1500):
            with open(PATH+directory+'/trajectory-'+str(trajectory_num)+'.p', 'rb') as f:
                trajectory = pickle.load(f)
                if len(trajectory) > maximum_length:
                    maximum_length = len(trajectory)
    print('Max length: ', maximum_length)
    
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
        
def panacea(K, datasets, I_min, I_max, D_safety, pi_e, pi_b, J_pi_b): 
    print('Computing results for Panacea...')
    with open(PATH+'results/with_panacea.csv', 'a') as f:
        for directory in datasets:
            for weighting in ['IS', 'WIS']:
                for k in K[1:]:
                    if weighting == 'IS':
                        Alpha = ALPHA_IS
                    else:
                        Alpha = ALPHA_WIS
                    for alpha in Alpha:
                        if weighting == 'IS':
                            c = compute_c('CH', 'IS', alpha, k, 1500)
                        else: 
                            c = compute_c('CH', 'WIS', alpha, k, 1500, I_min)
                        if c > 0:
                            J_hat_pi_e = adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, DELTA, PATH+'data/grid_b_policy/'+directory+'/', weighting, k, I_max, True, c)
                        
                            f.write(str(k)+'\t'+weighting+'\t'+str(alpha)+'\t'+str(c)+'\t'+str(J_hat_pi_e)+'\tGrid-world\n')
                            f.flush()
    
def safety_tests(K, datasets, I_min, I_max, D_safety, pi_e, pi_b, J_pi_b):
    print('Computing results for safety tests...')
    with open(PATH+'results/without_panacea.csv', 'a') as f:
        for directory in datasets:
            for weighting in ['IS', 'WIS']:
                for k in K:
                    J_hat_pi_e = adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, DELTA, PATH+'data/grid_b_policy/'+directory+'/', weighting, k, I_max)
                    f.write(str(k)+'\t'+weighting+'\t'+str(J_hat_pi_e)+'\tGrid-world\n')
                    f.flush()  
 
def run_grid(arg):  
    if arg == '1': # pick random behavior and evaluation policy
        print('Behavior policy selected.')
        pi_b = return_rand_policy()
        print('Creating datasets for behaviour policy...')
        create_datasets(pi_b)
        print('Computing performance of behaviour policy...')
        J_pi_b = compute_J(pi_b)
        print('J(\pi_b): ', J_pi_b)
        print('Finding a suitable evaluation policy...')
        flag = True
        while flag:
            pi_e = return_rand_policy() # evalution policy 
            J_pi_e = compute_J(pi_e)
            if J_pi_e < J_pi_b:
                flag = False 
        print('Evaluation policy selected.')
        print('J(\pi_e): ', J_pi_e)
    else:
        print('Behavior policy selected.')
        pi_b = create_pi_b()
        print('Creating datasets for behaviour policy...')
        #create_datasets(pi_b)
        print('Computing performance of behaviour policy...')
        #J_pi_b = compute_J(pi_b)
        J_pi_b = 0.7975473259641569#0.7970902655221709
        print('J(\pi_b): ', J_pi_b)
        pi_e = create_pi_e()
        print('Computing performance of evaluation policy...')
        #J_pi_e = compute_J(pi_e)
        J_pi_e = 0.7327537683014396#0.7280028364424095
        print('J(\pi_e): ', J_pi_e)
    
    # Datasets of behavior policy
    filenames = os.listdir(PATH+'data/grid_b_policy/')
    datasets = [filename for filename in filenames if filename.startswith('trajectories')]
    
    I_min = min_weight(pi_e, pi_b)
    I_max = max_weight(pi_e, pi_b)
    K = np.arange(0, 151, 1)
    D_safety = np.arange(0, NUM_POINTS_PER_DATASET)
    
    safety_tests(K, datasets, I_min, I_max, D_safety, pi_e, pi_b, J_pi_b)
    panacea(K, datasets, I_min, I_max, D_safety, pi_e, pi_b, J_pi_b)
    return J_pi_b, J_pi_e
    
if __name__=="__main__":
    main()
