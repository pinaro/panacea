#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:23:22 2020

@author: Pinar
@description: running experiments for testing
"""

from spi import *
from tamper_adversarially import *
from grid import *
from constants import *
import shutil
import random
import string
import os
import pickle
from decimal import *
from os import listdir
from os.path import isfile, join

TIME_STEPS = 100000

def calculate_TAU():
    tau = -1
    for i in range(0, 10):
        directory, tmp = sample_trajectories(return_rand_policy(), False)
        if tmp > tau:
            tau = tmp
    assert tau > -1
    return tau

def calculate_B():
    max_ratio = -1
    for i in range(0, TIME_STEPS):
        pi_b = return_rand_policy()
        pi_e = return_rand_policy()
        ratio = create_adversarial_trajectory(pi_e, pi_b)
        if(ratio > max_ratio):
            max_ratio = ratio
    assert max_ratio > -1
    return max_ratio
                    
def generate_rand_policies():
    new_directory = PATH+'policies/'
    os.mkdir(new_directory)
    for i in range(0, TIME_STEPS):
        pi_b = return_rand_policy()
        J_pi_b = compute_J(pi_b)
        pickle.dump(pi_b, open(new_directory+'policy-'+str(J_pi_b)+'.p', 'wb'))
    return new_directory

#def main():
def find_min_ratio():
    min_ratio = 1e+100
    onlyfiles = [f for f in listdir(PATH+'policies-tau=10/') if isfile(join(PATH+'policies-tau=10/', f))]
    for i in range(0, len(onlyfiles)):
        j_pi_e = float(onlyfiles[i].split('-')[1].split('.p')[0])
        for j in range(0, len(onlyfiles)):
            j_pi_b = float(onlyfiles[j].split('-')[1].split('.p')[0])
            if j_pi_e < j_pi_b:
                with open(join(PATH+'policies-tau=10/', onlyfiles[i]), 'rb') as f1:
                    with open(join(PATH+'policies-tau=10/', onlyfiles[j]), 'rb') as f2:
                        pi_e = pickle.load(f1) 
                        pi_b = pickle.load(f2)
                        ratio = create_adversarial_trajectory(pi_e, pi_b)
                        if ratio < min_ratio:
                            min_ratio = ratio
                            pi_e_file = onlyfiles[i]
                            pi_b_file = onlyfiles[j]
    print('min ratio', min_ratio)
    print('pi_e filename: ', pi_e_file)
    print('pi_b filename: ', pi_b_file) 
 
# Use good policy as tester
def create_pi_b():
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

def main_tmp():
#def main():
    #print('Calculating TAU...')
    #tau = calculate_TAU()
    #print('TAU: ', tau)
    #print('Calculating B...')
    #b = calculate_B()
    #print('B: ', b)
    generate_rand_policies()    

def create_datasets(): 
    pi_b = create_pi_b()
    m = -10
    for i in range(0, 750):
        directory, max_time = sample_trajectories(pi_b, False, 1500)
        if max_time > m:
            m = max_time
    print('Max time: ', m)      

def main():
    # List directories
    filenames = os.listdir(PATH)
    datasets = [filename for filename in filenames if filename.startswith('trajectories')]
    # Behavior policy
    K = np.arange(0, 151, 1)
    pi_b = create_pi_b()
    J_pi_b = 0.7975473259641569#compute_J(pi_b)
    print('J_pi_b: ', J_pi_b)
    # Evaluation policy
    pi_e = create_pi_e()
    J_pi_e = 0.7327537683014396#compute_J(pi_e)
    print('J_pi_e: ', J_pi_e)
    
    D_safety = np.arange(0, 1500, 1)
    delta = DELTA
    w_max = adversarially_monotonic_trajectory(pi_e, pi_b)
    
    with open('/Users/Pinar/Desktop/NeurIPS_fig1/grid_results.csv', 'w+') as f:
        f.write('k\tEstimator\tResult\tProblem\n')
        for directory in datasets:
            for CI in ['CH']:
                for weighting in ['IS', 'WIS']:
                    for k in K:
                        if CI == 'AM' and weighting == 'WIS':
                            w_min = AM_WIS_trajectory(pi_e, pi_b)
                            J_hat_pi_e_min = adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, PATH+directory+'/', weighting, CI, k, w_min)
                            J_hat_pi_e_max = adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, PATH+directory+'/', weighting, CI, k, w_max)
                            if J_hat_pi_e_min > J_hat_pi_e_max: 
                                J_hat_pi_e = J_hat_pi_e_min
                            else: 
                                J_hat_pi_e = J_hat_pi_e_max
                        else:
                            J_hat_pi_e = adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, PATH+directory+'/', weighting, CI, k, w_max)
                        f.write(str(k)+'\t'+weighting+'\t'+str(J_hat_pi_e)+'\tGrid-world\n')
                        f.flush()                    
if __name__=="__main__":
    main()
    
#def main():
##def main_tmp():
#    delta = DELTA
#    k = 8500
#    inc = np.arange(500,551, 1)
#    with open(PATH+'policies/policy-0.21117261055714404.p', 'rb') as f1:
#        #pi_b = pickle.load(f1)
#        #J_pi_b = 0.2320250332314022
#        pi_b = create_pi_b()
#        J_pi_b = compute_J(pi_b)
#        with open(PATH+'policies/policy-0.20916456830609007.p', 'rb') as f2:
#            #pi_e = pickle.load(f2)  
#            #J_pi_e = 0.2682965253335817
#            pi_e = create_pi_e()
#            J_pi_e = compute_J(pi_e)
#            #directory, max_time = sample_trajectories(pi_b, True)
#            directory = PATH+'trajectories-WUJUIPDiks/'
#            
#            #print('Max time: ', max_time)
#            #print('Tau: ', TAU)
#            #assert TAU > max_time
#            w = adversarially_monotonic_trajectory(pi_e, pi_b)
#              
#            with open(PATH+'fabricated-trajectories-brr.csv', 'w+') as f:
#                f.write('K\tCI\tWeighting\tJ_pi_b\tJ_pi_e\tJ_hat_pi_e\n')
#                for CI in ['CH', 'AM']:
#                    for weighting in ['IS', 'WIS']:
#                        if CI == 'AM' and weighting == 'WIS':
#                            w = AM_WIS_trajectory(pi_e, pi_b)
#                        for i in inc:#range(0, k+1): # number of trajectories added inc:
#                            J_hat_pi_e = adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, directory, weighting, CI, i, w)
#                            f.write(str(i)+'\t'+CI+'\t'+weighting+'\t'+str(J_pi_b)+'\t'+str(J_pi_e)+'\t'+str(J_hat_pi_e)+'\n')
#                            f.flush()  

# A very good policy has J(pi) around 0.7426467858737733 for grid world 
    
# =============================================================================
# def create_pi_e():
#     pi_e = np.full((ROW*COL, len(ACTIONS)), 0.25)
#     
#     pi_e[0, 2] = 0.7 # right at state 0
#     pi_e[0, 0] = 0.1
#     pi_e[0, 1] = 0.1
#     pi_e[0, 3] = 0.1
#     
#     pi_e[3, 2] = 0.7 # right at state 3
#     pi_e[3, 0] = 0.1
#     pi_e[3, 1] = 0.1
#     pi_e[3, 3] = 0.1
#     
#     pi_e[6, 1] = 0.7 # down at state 6
#     pi_e[6, 2] = 0.1
#     pi_e[6, 3] = 0.1
#     pi_e[6, 0] = 0.1
#     
#     pi_e[7, 2] = 0.11 # down at state 7
#     pi_e[7, 0] = 0.11
#     pi_e[7, 3] = 0.11
#     pi_e[7, 1] = 1-0.33
#     
#     #pi_e[7, 2] = 0.14 # down at state 7
#     #pi_e[7, 0] = 0.13
#     #pi_e[7, 3] = 0.13
#     #pi_e[7, 1] = 0.6
#     
#     pi_e[8, 1] = 0.7 # down at state 8
#     pi_e[8, 2] = 0.1
#     pi_e[8, 3] = 0.1
#     pi_e[8, 0] = 0.1
#     return pi_e
# =============================================================================

# =============================================================================
# def create_pi_b():
#     pi_e = np.full((ROW*COL, len(ACTIONS)), 0.25)
#     
#     pi_e[0, 2] = 0.7 # right at state 0
#     pi_e[0, 0] = 0.1
#     pi_e[0, 1] = 0.1
#     pi_e[0, 3] = 0.1
#     
#     pi_e[3, 2] = 0.7 # right at state 3
#     pi_e[3, 0] = 0.1
#     pi_e[3, 1] = 0.1
#     pi_e[3, 3] = 0.1
#     
#     pi_e[6, 1] = 0.7 # down at state 6
#     pi_e[6, 2] = 0.1
#     pi_e[6, 3] = 0.1
#     pi_e[6, 0] = 0.1
#     
#     pi_e[7, 2] = 0.1 # down at state 7
#     pi_e[7, 0] = 0.1
#     pi_e[7, 3] = 0.1
#     pi_e[7, 1] = 0.7
#     
#     pi_e[8, 1] = 0.7 # down at state 8
#     pi_e[8, 2] = 0.1
#     pi_e[8, 3] = 0.1
#     pi_e[8, 0] = 0.1
#     return pi_e
# =============================================================================
#def create_pi_b():
#    pi_e = np.full((ROW*COL, len(ACTIONS)), 0.25)
#    
#    pi_e[0, 2] = 0.7 # right at state 0
#    pi_e[0, 0] = 0.1
#    pi_e[0, 1] = 0.1
#    pi_e[0, 3] = 0.1
# 
#    pi_e[3, 2] = 0.1 # down at state 3
#    pi_e[3, 0] = 0.1
#    pi_e[3, 1] = 0.7
#    pi_e[3, 3] = 0.1
#    
#    pi_e[4, 1] = 0.1 # right at state 4
#    pi_e[4, 2] = 0.7
#    pi_e[4, 3] = 0.1
#    pi_e[4, 0] = 0.1
#    
#    pi_e[7, 2] = 0.2 # down at state 7
#    pi_e[7, 0] = 0.2
#    pi_e[7, 1] = 0.5
#    pi_e[7, 3] = 0.1
#    
#    return pi_e
#
#
#def create_pi_e():
#    pi_e = np.full((ROW*COL, len(ACTIONS)), 0.25)
#
#    pi_e[0, 2] = 0.7 # right at state 0
#    pi_e[0, 0] = 0.1
#    pi_e[0, 1] = 0.1
#    pi_e[0, 3] = 0.1
# 
#    pi_e[3, 2] = 0.1 # down at state 3
#    pi_e[3, 0] = 0.1
#    pi_e[3, 1] = 0.7
#    pi_e[3, 3] = 0.1
#    
#    pi_e[4, 1] = 0.15 # right at state 4
#    pi_e[4, 2] = 0.65
#    pi_e[4, 3] = 0.1
#    pi_e[4, 0] = 0.1
#    
#    pi_e[7, 2] = 0.2 # down at state 7
#    pi_e[7, 0] = 0.2
#    pi_e[7, 1] = 0.5
#    pi_e[7, 3] = 0.1
#
#    return pi_e