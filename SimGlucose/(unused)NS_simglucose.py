from Environments.SimGlucose.simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from Environments.SimGlucose.simglucose.patient.t1dpatient import T1DPatient
from Environments.SimGlucose.simglucose.sensor.cgm import CGMSensor
from Environments.SimGlucose.simglucose.actuator.pump import InsulinPump
from Environments.SimGlucose.simglucose.simulation.scenario_gen import RandomScenario
from Environments.SimGlucose.simglucose.controller.base import Action

import pandas as pd
import numpy as np
from datetime import datetime
from Src.Utils.utils import Space
from os import path

curr_path = path.abspath(path.join(path.dirname(__file__)))
PATIENT_PARA_FILE = path.join(curr_path, 'simglucose', 'params', 'vpatient_params.csv')


class T1DSimEnv(object):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 speed=1,
                 patient_name=None,
                 reward_fun=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        self._seed(0)

        self.speed = speed
        hour = self.rng.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = 'adolescent#001'

        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=0)
        pump = InsulinPump.withName('Insulet')
        scenario = RandomScenario(start_time=start_time, seed=0)
        #
        self.env = _T1DSimEnv(patient, sensor, pump, scenario)

        self.reward_fun = reward_fun

    @staticmethod
    def pick_patient():
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        while True:
            print('Select patient:')
            for j in range(len(patient_params)):
                print('[{0}] {1}'.format(j + 1, patient_params['Name'][j]))
            try:
                select = int(input('>>> '))
            except ValueError:
                print('Please input a number.')
                continue

            if select < 1 or select > len(patient_params):
                print('Please input 1 to {}'.format(len(patient_params)))
                continue

            return select

    def _step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        else:
            return self.env.step(act, reward_fun=self.reward_fun)

    def _reset(self):
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def _render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return Space(low=0, high=ub, size=1)

    @property
    def observation_space(self):
        return Space(low=0, high=np.inf, size=1)


