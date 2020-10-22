#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:38:53 2020

@author: pinar
@description: Panacea: does it work?
"""

from spi import *
from constants import *
from grid import *
from tamper_adversarially import *
import numpy as np
import shutil

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
    
def split_data():
    filenames = os.listdir(PATH)
    datasets = [filename for filename in filenames if filename.startswith('trajectories')]
    i = 1
    j = 1
    for directory in datasets:
        shutil.move(PATH+directory+'/', '/Users/Pinar/Desktop/NeurIPS_fig1/set'+str(i)+'/')
        if j % 75 == 0:
            i += 1
        j += 1
    
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

def main():
    # List directories
    num = 10
    p = '/Users/Pinar/Desktop/NeurIPS_fig1/grid_b_policies/set' + str(num) + '/'
    filenames = os.listdir(p)
    datasets = [filename for filename in filenames if filename.startswith('trajectories')]
    
    delta = DELTA
    K = np.arange(1, 151, 1)
    pi_b = create_pi_b()
    pi_e = create_pi_e()
    D_safety = np.arange(0, 1500)
    
    I_min = AM_WIS_trajectory(pi_e, pi_b)
    I_max = adversarially_monotonic_trajectory(pi_e, pi_b)
    #print('I_min: ', I_min)
    #print('I_max: ', I_max)
    Alpha_WIS = [0.01, 0.5, 1]#[0.1, 0.5, 1]
    Alpha_IS = [0.01, 0.1, 0.5, 1, 5]#[1, 5]#[0.1, 0.5, 1, 5]
    #J_pi_b = compute_J(pi_b)
    #print('J_pi_b: ', J_pi_b)
    #J_pi_e = compute_J(pi_e)
    #print('J_pi_e: ', J_pi_e)
    J_pi_b = 0.7970902655221709
    J_pi_e = 0.7280028364424095
    
    filename = 'grid_panacea_' + str(num) + '.csv'
    with open('/Users/Pinar/Desktop/NeurIPS_fig1/'+filename, 'w+') as f:
        f.write('k\tEstimator\tpanaceaAlpha\tClip\tResult\tProblem\n')
        for directory in datasets:
            for CI in ['CH']:
                for weighting in ['IS', 'WIS']:
                    for k in K:
                        if weighting == 'IS':
                            Alpha = Alpha_IS
                        else:
                            Alpha = Alpha_WIS
                        for alpha in Alpha:
                            if weighting == 'IS':
                                c = compute_c('CH', 'IS', alpha, k, 1500)
                            else: 
                                c = compute_c('CH', 'WIS', alpha, k, 1500, I_min)
                            assert c > 0
                            
                            J_hat_pi_e = adversarial_safety_test_panacea(pi_e, pi_b, D_safety, J_pi_b, delta, p+directory+'/', weighting, CI, k, I_max, c)
                            
                            f.write(str(k)+'\t'+weighting+'\t'+str(alpha)+'\t'+str(c)+'\t'+str(J_hat_pi_e)+'\tGrid-world\n')
                            f.flush()
    
if __name__=="__main__":
    main()
    