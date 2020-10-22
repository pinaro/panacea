#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:50:24 2020

@author: pinar
@description: Panacea: does it work?
"""

from spi import *
from constants import *
#from panacea import *
from grid import *
from tamper_adversarially import *
import numpy as np

TOTAL = [100000]

def create_pi_b():
    pi_e = np.full((ROW*COL, len(ACTIONS)), 0.25)
    
    pi_e[0, 2] = 0.7 # right at state 0
    pi_e[0, 0] = 0.1
    pi_e[0, 1] = 0.1
    pi_e[0, 3] = 0.1
 
    pi_e[3, 2] = 0.1 # down at state 3
    pi_e[3, 0] = 0.1
    pi_e[3, 1] = 0.7
    pi_e[3, 3] = 0.1
    
    pi_e[4, 1] = 0.1 # right at state 4
    pi_e[4, 2] = 0.7
    pi_e[4, 3] = 0.1
    pi_e[4, 0] = 0.1
    
    pi_e[7, 2] = 0.2 # down at state 7
    pi_e[7, 0] = 0.2
    pi_e[7, 1] = 0.5
    pi_e[7, 3] = 0.1
    return pi_e

def create_pi_e():
    pi_e = np.full((ROW*COL, len(ACTIONS)), 0.25)

    pi_e[0, 2] = 0.7 # right at state 0
    pi_e[0, 0] = 0.1
    pi_e[0, 1] = 0.1
    pi_e[0, 3] = 0.1
 
    pi_e[3, 2] = 0.1 # down at state 3
    pi_e[3, 0] = 0.1
    pi_e[3, 1] = 0.7
    pi_e[3, 3] = 0.1
    
    pi_e[4, 1] = 0.15 # right at state 4
    pi_e[4, 2] = 0.65
    pi_e[4, 3] = 0.1
    pi_e[4, 0] = 0.1
    
    pi_e[7, 2] = 0.2 # down at state 7
    pi_e[7, 0] = 0.2
    pi_e[7, 1] = 0.5
    pi_e[7, 3] = 0.1
    return pi_e

def g(n, i):
    tmp = np.log(2/DELTA) / (2*n)
    tmp = np.sqrt(tmp)
    result = (i / n) + tmp
    return np.minimum(1, result)

def estimate_alpha(alpha_directory, k, I_max, D_safety, weighting, CI, pi_e, pi_b, J_pi_b, without, delta=DELTA):
    with_adversary = adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, alpha_directory, weighting, CI, k, I_max)
    return with_adversary - without
    
def compute_c(alpha_hat, n, k, i_max, weighting, CI, pi_e=None, pi_b=None, extra_data_CH_WIS=None):
    if CI == 'CH' and weighting == 'IS':
        tmp = alpha_hat * (n + k)
        return i_max - (tmp / k)
    elif CI == 'AM' and weighting == 'IS':
        total = 0
        for i in range(n+1, n+k):
            total += (g(n+k, i) - g(n+k, i-1))
        print('IN FUNCTION')
        print('TOTAL: ', total)
        if total == 0:
            return 0
        else:
            c = i_max - (alpha_hat / total) 
            if c <= 0:
                return 0
            else:
                return c
    elif CI == 'CH' and weighting == 'WIS':
        D_safety = np.arange(0, n)
        is_weights, rewards = importance_sampling_estimates_panacea(pi_e, pi_b, D_safety, extra_data_CH_WIS)    
        total = np.sum(is_weights * rewards)
        beta = np.sum(is_weights)
        one = k * i_max * beta
        two = k * i_max * total
        three = alpha_hat * beta * k * i_max
        four = alpha_hat * (beta**2)
        numerator = one - two - three - four
        one = k * alpha_hat * i_max
        two = alpha_hat * beta
        denominator = (one + two + beta - total) * k 
        return numerator / denominator

def main():
    delta = DELTA
    increments = 10 # how often to calculate k
    K = np.arange(0, 8500, 100)
    K[0] = 1
    pi_b = create_pi_b()
    pi_e = create_pi_e()
    J_pi_b = 0.534411498824547#compute_J(pi_b)
    I_min = AM_WIS_trajectory(pi_e, pi_b)
    I_max = adversarially_monotonic_trajectory(pi_e, pi_b)
    main_directory = PATH+'trajectories-WUJUIPDiks/'
    extra_data_CH_WIS = PATH+'trajectories-Jw3ia693gg/'
    dirs = [PATH+'trajectories-IcRr1uRQAb/', PATH+'trajectories-09GDtb8YBa/', PATH+'trajectories-LkHjGasp3m/', PATH+'trajectories-XYAAsWANjp/', PATH+'trajectories-R5tw0mSQv1/', PATH+'trajectories-NzDFxKkHfM/', PATH+'trajectories-l3Ytz5l49m/', PATH+'trajectories-JDPKISRIxA/', PATH+'trajectories-7925fx5elg/', PATH+'trajectories-Jw3ia693gg/']
    for i in range(0, 10):
        directory, max_time = sample_trajectories(pi_b, True, TOTAL[0])
        dirs.append(directory)
    with open(PATH+'panacea-solve-c.csv', 'w+') as f:
        f.write('n\tk\tc\tCI\tWeighting\tJ_hat_pi_e\n')
        for n in TOTAL:
            D_safety = np.arange(0, n)
            #alpha_directory = PATH+'trajectories-O4wUjmcwhz/'
            for CI in ['CH', 'AM']:
                for weighting in ['WIS']:
                    if CI != 'AM' or weighting != 'WIS':
                        for k in K:
                            total_c = 0
                            for directory in dirs:
                                without = adversarial_safety_test(pi_e, pi_b, D_safety, J_pi_b, delta, directory, weighting, CI, 0, I_max)
                                alpha_hat = estimate_alpha(directory, k, I_max, D_safety, weighting, CI, pi_e, pi_b, J_pi_b, without)
                                c = compute_c(alpha_hat, n, k, I_max, weighting, CI, pi_e, pi_b, extra_data_CH_WIS)
                                total_c += c
                            c = total_c / len(dirs) 
                            J_hat_pi_e = adversarial_safety_test_panacea(pi_e, pi_b, D_safety, J_pi_b, delta, main_directory, CI, weighting, k, I_max, c)
                            f.write(str(n)+'\t'+str(k)+'\t'+str(c)+'\t'+str(CI)+'\t'+str(weighting)+'\t'+str(J_hat_pi_e)+'\n')
    
if __name__=="__main__":
    main()
    