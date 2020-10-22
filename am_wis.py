#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:25:37 2020

@author: Pinar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:38:53 2020

@author: pinar
@description: max alpha-security of AM+WIS
"""

from spi import *
from constants import *
from panacea_solve_c import *
from grid import *
from tamper_adversarially import *
import numpy as np

TOTAL = [100, 1000, 10000]

def CH_IS(n, k, i_star):
    tmp1 = np.log(1/DELTA) / (2*n)
    m = n+k
    tmp2 = np.log(1/DELTA) / (2*m)
    term1 = np.sqrt(tmp1) - np.sqrt(tmp2) + (k/m)
    return i_star * term1

def CH_WIS(n, k, i_star, i_min):
    tmp = np.log(1/DELTA) / (2*n)
    term1 = np.sqrt(tmp)
    m = n+k
    tmp = np.log(1/DELTA) / (2*m)
    term2 = np.sqrt(tmp)
    tmp = (k * i_star)
    term3 = tmp / (i_min + tmp)
    return term1 - term2 + term3#+ 1

def g(n, i):
    tmp = np.log(2/DELTA) / (2*n)
    tmp = np.sqrt(tmp)
    result = (i / n) + tmp
    return np.minimum(1, result)

def term1_AM_IS(n, k, i_star):
    total = 0
    for i in range(n, n+k):
        total += (g(n+k, i) - g(n+k, i-1))
    return i_star * total

def term2_AM_IS(n, k, i_star):
    t = g(n, 0) - g(n+k, 0) + g(n+k, n-1) - g(n, n-1)
    if t < 0:
        print('For AM+IS, 2nd term is negative.')
        return 0
    else:
        return i_star * t

def AM_IS(n, k, i_star):
    return term1_AM_IS(n, k, i_star) + term2_AM_IS(n, k, i_star)

def c_prime_CH_IS(alpha, n, k):
    tmp1 = np.log(1/DELTA) / (2*n)
    m = n+k
    tmp2 = np.log(1/DELTA) / (2*m)
    term1 = np.sqrt(tmp1) - np.sqrt(tmp2) + (k/m)
    return alpha / term1

def c_prime_CH_WIS(alpha, n, k, i_min):
    tmp = np.log(1/DELTA) / (2*n)
    term1 = np.sqrt(tmp)
    m = n+k
    tmp = np.log(1/DELTA) / (2*m)
    term2 = np.sqrt(tmp)
    numerator = i_min * (alpha - term1 + term2)
    tmp = 1-alpha
    denom = k * (tmp + term1 - term2)
    return numerator / denom
   
def c_prime_AM_IS(alpha, n, k):
    total = 0
    for i in range(n, n+k):
        total += (g(n+k, i) - g(n+k, i-1))
    t = g(n, 0) - g(n+k, 0) + g(n+k, n-1) - g(n, n-1)
    if t > 0:
        return alpha / (total + t)
    else:
        return alpha / total
            
def main():
    increments = 10 # how often to calculate k
    pi_b = create_pi_b()
    pi_e = create_pi_e()
    I_min = AM_WIS_trajectory(pi_e, pi_b)
    I_max = adversarially_monotonic_trajectory(pi_e, pi_b)
    with open('/Users/Pinar/Desktop/'+'max_alpha.csv', 'w+') as f:
        f.write('N\tK\tCI\tWeighting\tResult\tI_star\tC_prime\n')
        for n in TOTAL:
            K = np.arange(0, int(0.1*n)+1, increments)
            K[0] = 1
            for k in K: 
                result = CH_IS(n, k, I_max)
                c_prime = c_prime_CH_IS(result, n, k)
                flag = c_prime >= I_max
                f.write(str(n)+'\t'+str(k)+'\t'+'CH'+'\t'+'IS'+'\t'+str(result)+'\t'+str(I_max)+'\t'+str(c_prime)+'\t'+str(flag)+'\n')
                result = CH_WIS(n, k, I_max, I_min)
                c_prime = c_prime_CH_WIS(result, n, k, I_min)
                flag = c_prime >= I_max
                f.write(str(n)+'\t'+str(k)+'\t'+'CH'+'\t'+'WIS'+'\t'+str(result)+'\t'+str(I_max)+'\t'+str(c_prime)+'\t'+str(flag)+'\n')
                result = AM_IS(n, k, I_max)
                c_prime = c_prime_AM_IS(result, n, k)
                flag = c_prime >= I_max
                f.write(str(n)+'\t'+str(k)+'\t'+'AM'+'\t'+'IS'+'\t'+str(result)+'\t'+str(I_max)+'\t'+str(c_prime)+'\t'+ str(flag)+'\n')
                #result = AM_WIS(n, k, I_min, I_max)
                #f.write(str(n)+'\t'+str(k)+'\t'+'AM'+'\t'+'WIS'+'\t'+str(result)+'\n')
                f.flush()
    
if __name__=="__main__":
    main()
    
# Test to see if max is calculate incorrectly
#def term1_AM_IS_v2(n, k, i_star):
#    total = 0
#    for i in range(n, n+k):
#        total += (g(n+k, i) - g(n+k, i-1))
#    return i_star * total
#
#def term2_AM_IS_v2(n, k, i_star):
#    base = i_star - 1
#    inc = 1 / n
#    total = 0
#    for i in range(1, n):
#        multiplier = ((i+1) * inc) + base
#        if i == n-1:
#            assert multiplier == i_star
#        print(multiplier)
#        total += (g(n+k, i) - g(n+k, i-1) + g(n, i-1) - g(n, i)) * multiplier
#    return total
#
#def AM_IS_v2(n, k, i_star):
#    return term1_AM_IS_v2(n, k, i_star) + term2_AM_IS_v2(n, k, i_star)
#
#def term1_AM_WIS(n, k, i_min, i_max):
#    return g(n, 0) - g(n, n-1)
#
#def term2_AM_WIS(n, k, i_min, i_max):
#    m = n+k
#    tmp1 = g(n+k, n+k-1) - g(n+k, k)
#    total = 0
#    for i in range(1, k+1):
#        total += (g(n+k, i) - g(n+k, i-1))
#    two = tmp1 + total
#    denom = (k * i_min) + (n * i_max)
#    mult1 = (m * i_max) / denom
#    term1 = mult1 * tmp1
#    mult2 = (m * i_min) / denom
#    term2 = mult2 * total
#    one = term1 + term2 
#    if one > two:
#        return one
#    else:
#        #print('No multiplier')
#        return two
#    
#def AM_WIS(n, k, i_min, i_max):
#    return term1_AM_WIS(n, k, i_min, i_max) + term2_AM_WIS(n, k, i_min, i_max)

#def term1_AM_WIS_v2(n, k, i_min, i_max):
#    return g(n, 0) - g(n, n-1)
#
#def term2_AM_WIS_v2(n, k, i_min, i_max):
#    m = n+k
#    total = 0
#    for i in range(1, k+1):
#        total += (g(n+k, i) - g(n+k, i-1))
#    return (m / (k)) * total
#
#def term2_AM_WIS_v2(n, k, i_min, i_max):
#    m = n+k
#    tmp = g(m, m-1) - g(m, k)
#    return (m / n) * tmp
#    
#def AM_WIS_v2(n, k, i_min, i_max):
#    return term1_AM_WIS_v2(n, k, i_min, i_max) + term2_AM_WIS_v2(n, k, i_min, i_max)
    