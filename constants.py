# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:52:39 2017

@author: Pinar Ozisik
@description: Constants for safe policy improvement
"""

# How many squares are there in the grid
ROW = 3
COL = 3

ACTIONS = ['UP', 'DOWN', 'RIGHT', 'LEFT']

START = (0, 0) # start state
FINISH = (ROW-1, COL-1) # end state

EPISODES = 100000 #split between safety and candidate data -- going more than 1e+5 takes forever
TRIALS = 10000#int(1e+6) # number of trials for computing J(policy)
SPI_TIME = 20 # 50 rounds of SPI
PATH = '/Users/Pinar/Desktop/NeurIPS_fig1/grid_b_policies/' #/mnt/nfs/work1/brian/pinar/Safe-Secure-RL/

# Hyperparameters
GAMMA = 0.95
SIGMA = 0.5

# Portion of D to split into D_candidate and D_safety
SPLIT = 1 # safety get 80% candidate gets 20%
DELTA = 0.05

# CMA-ES parameters
SIGMA = 1.0
MAXITER = 50
TOLFUN = 1e-12
DISCOUNT_FACTOR = 10
#CMA_TRIALS = 5000 # number of trials for computing J(policy) in CMA-ES

# Adversarial Parameters
COUNT = 1
BIG_CONSTANT = 1000000

# Safe RL parameters
B = 60#30
TAU = 51 #10 # can't get it to handle more than 20 for AM. Can handle anything for CH, but setting B is an issue 
# AM got it to work for TAU = 10 both for IS and WIS