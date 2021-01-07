# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:52:39 2017

@author: Pinar Ozisik
@description: Constants
"""

# Grid constants
ROW = 3 # rows and cols in the grid
COL = 3
ACTIONS = ['UP', 'DOWN', 'RIGHT', 'LEFT']
START = (0, 0) # start state
FINISH = (ROW-1, COL-1) # end state
GAMMA = 0.95 # discount factor
TAU = 51 #<-- was set to this during results #10 # can't get it to handle more than 20 for AM. Can handle anything for CH, but setting B is an issue 

# Diabetes constants
PATIENT = 'adult#003' # Found patient-specific variables from /params/Quest.csv
CR_MEAN = 9
CF_MEAN = 17.9345522688
MIN_RETURN = -45
MAX_RETURN = -20
CR_LOW, CF_LOW = 3, 5 #env.action_space.low
CR_HIGH, CF_HIGH = 30, 50 #env.action_space.high

# Constants for both domains
PATH = './'
DELTA = 0.05
NUM_DATASETS = 100#750
NUM_POINTS_PER_DATASET = 1000#1500
NUM_SAMPLES_FOR_J = 10000
ALPHA_IS = [0.1, 0.5, 1, 5]
ALPHA_WIS = [0.01, 0.5, 1]