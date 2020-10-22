#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:38:53 2020

@author: pinar
@description: alpha-security of current methods
"""

from spi import *
from constants import *
from panacea_solve_c import *
from grid import *
from tamper_adversarially import *
import numpy as np

TOTAL = [100]#[100, 1000, 10000]

def CH_IS(n, k, i_star):
    tmp = np.log(1/DELTA) / (2*n)
    term1 = i_star * np.sqrt(tmp)
    m = n+k
    tmp = np.log(1/DELTA) / (2*m)
    term2 = i_star * np.sqrt(tmp)
    return term1 - term2

def CH_WIS(n, k, i_star):
    tmp = np.log(1/DELTA) / (2*n)
    term1 = np.sqrt(tmp)
    m = n+k
    tmp = np.log(1/DELTA) / (2*m)
    term2 = np.sqrt(tmp)
    return term1 - term2

def g(n, i):
    tmp = np.log(2/DELTA) / (2*n)
    tmp = np.sqrt(tmp)
    result = (i / n) + tmp
    return np.minimum(1, result)

def term1_AM_IS(n, k, i_star):
    total = 0
    for i in range(n+1, n+k):
        total += (g(n+k, i) - g(n+k, i-1))
    return i_star * total

def term2_AM_IS(n, k, i_star):
    multiplier = np.zeros(n)
    for i in range(1, n):
        multiplier[i-1] = g(n+k, i) - g(n+k, i-1) + g(n, i-1) - g(n, i)
    multiplier[n-1] = g(n+k, n) - g(n+k, n-1)
    t = np.sum(multiplier)
    if t > 0:
        return 0
    else:
        return i_star * t

def AM_IS(n, k, i_star):
    return term1_AM_IS(n, k, i_star) + term2_AM_IS(n, k, i_star)

def term1_AM_WIS(n, k, i_min, i_max):
    total = 0
    for i in range(1, k+1):
        total += (g(n+k, i) - g(n+k, i-1))
    num = i_min * total * (n+k)
    denom = (k * i_min) + (n * i_max)
    return num / denom

def term2_AM_WIS(n, k, i_min, i_max):
    num = i_max * (n+k)
    denom = (k * i_min) + (n * i_max) 
    tmp = num / denom
    multiplier = np.zeros(n-1)
    for i in range(1, n):
        one = g(n+k, i+k) - g(n+k, i+k-1)
        two = g(n, i-1) - g(n, i)
        multiplier[i-1] = (tmp * one) + two
    t = np.sum(multiplier)
    if t > 0:
        return 0
    else:
        return t
    
def AM_WIS(n, k, i_min, i_max):
    return term1_AM_WIS(n, k, i_min, i_max) + term2_AM_WIS(n, k, i_min, i_max)

def main():
    increments = 10 # how often to calculate k
    pi_b = create_pi_b()
    pi_e = create_pi_e()
    I_min = AM_WIS_trajectory(pi_e, pi_b)
    I_max = adversarially_monotonic_trajectory(pi_e, pi_b)
    with open(PATH+'min_alpha_xtra.csv', 'w+') as f:
        f.write('N\tK\tCI\tWeighting\tResult\n')
        for n in TOTAL:
            K = np.arange(0, int(0.1*n)+1, 1)#increments)
            K[0] = 1
            for k in K: 
                result = CH_IS(n, k, I_max)
                f.write(str(n)+'\t'+str(k)+'\t'+'CH'+'\t'+'IS'+'\t'+str(result)+'\n')
                result = CH_WIS(n, k, I_max)
                f.write(str(n)+'\t'+str(k)+'\t'+'CH'+'\t'+'WIS'+'\t'+str(result)+'\n')
                result = AM_IS(n, k, I_max)
                f.write(str(n)+'\t'+str(k)+'\t'+'AM'+'\t'+'IS'+'\t'+str(result)+'\n')
                result = AM_WIS(n, k, I_min, I_max)
                f.write(str(n)+'\t'+str(k)+'\t'+'AM'+'\t'+'WIS'+'\t'+str(result)+'\n')
                f.flush()
    
if __name__=="__main__":
    main()
    