#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:47:38 2020

@author: pinar
@description: 
"""

import sys
import os
import subprocess
from constants import *
from run_diabetes import *

# Takes an array of floats and coverts it into a form that can be used in R
def build_Rlist(array):
    string = ''
    for item in array:
        string += str(item)+','
    return string[:-1] 

def main():
    #os.mkdir(PATH+'data/')
    #os.mkdir(PATH+'results/')
    
    print('Starting experiments for the diabetes domain.')
    J_b_diabetes, J_e_diabetes = run_diabetes(sys.argv[1])
    #J_b_diabetes, J_e_diabetes = 0.21880416377956427, 0.14524769684920283
    #print('Starting experiments for grid-world.')
    #J_b_grid, J_e_grid = run_grid(sys.argv[1])
    #J_b_grid, J_e_grid = 0.7970902655221709, 0.7280028364424095
    print('Plotting results...')
    
    IS_lst = build_Rlist(ALPHA_IS)
    WIS_lst = build_Rlist(ALPHA_WIS)
    #command = 'rscript /Users/pinar/Desktop/panacea_code_to_publish/plot_results.R'
    command = 'Rscript plot_results.R '+str(J_b_diabetes)+' '+str(J_e_diabetes)+' '+str(J_b_grid)+' '+str(J_e_grid)+' '+IS_lst+' '+WIS_lst
    print(command)
    output = subprocess.check_output(command, shell=True)
    #print('Created results.pdf in the project folder.')
    
if __name__=="__main__":
    main()