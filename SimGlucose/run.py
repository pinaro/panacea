#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:57:19 2020

@author: Pinar
@description: Code for NeurIPS paper
"""

#import gym
#from gym.envs.registration import register
#
## Register gym environment. By specifying kwargs,
## you are able to choose which patient to simulate.
## patient_name must be 'adolescent#001' to 'adolescent#010',
## or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
#
#register(
#    id='NS_SimGlucose-v0',
#    entry_point='simglucose.envs:T1DSimEnv',
#    kwargs={'patient_name': 'adolescent#002'})
#
#
#env = gym.make('NS_SimGlucose-v0')
#
#observation = env.reset()
##CR, CF = (4, 15)
##action = (4, 15)
#for t in range(100):
#    #env.render(mode='human')
#    #print(observation)
#    # Action in the gym environment is a scalar
#    # representing the basal insulin, which differs from
#    # the regular controller action outside the gym
#    # environment (a tuple (basal, bolus)).
#    # In the perfect situation, the agent should be able
#    # to control the glucose only through basal instead
#    # of asking patient to take bolus
#    #action = env.action_space.sample()
#    observation, reward, done, info = env.step(action)
#    print('observation: ', observation)
#    print('reward: ', reward)
#    print('done:', done)
#    print()
#    #if done:
#    #    print("Episode finished after {} timesteps".format(t + 1))
#    #    break

import gym

from gym.envs.registration import register
register(
    id='Pinar_SimGlucose-v0',
    entry_point='simglucose.envs:Pinar_T1DSimEnv',
    kwargs={'patient_name':'adolescent#002'}
)

env = gym.make('Pinar_SimGlucose-v0')

observation = env.reset() # can reset only once
print('action space', env.action_space)
action = env.action_space.sample()
print('action', action)
for t in range(100):
    env.render(mode='human')
    print('observation', observation)
    # Action in the gym environment is a scalar
    # representing the basal insulin, which differs from
    # the regular controller action outside the gym
    # environment (a tuple (basal, bolus)).
    # In the perfect situation, the agent should be able
    # to control the glucose only through basal instead
    # of asking patient to take bolus
    observation, reward, done, info = env.step(action)
    print('reward', reward)
#    if done:
#        print("Episode finished after {} timesteps".format(t + 1))
#        break
    