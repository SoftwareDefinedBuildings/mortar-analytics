#TODO: add bounds for temperature setpoints

from pyfmi import load_fmu
import yaml
import numpy as np
import asyncio

from flask import Flask, jsonify
import threading

import datetime
import pytz
import pandas as pd

## TODO: include get_current_state(): both fast system and slower radiant system
## TODO: logging functions
## TODO: set initial setpoint to minimum setpoint --> currently in boiler CDL -> t&R block; not exposed. ask michael

class Boiler_Controller:
    def __init__(self, config_file='config.yaml'):
        with open(config_file) as fp:
            self.config = yaml.safe_load(fp)

        self._default_setpoint = self.config.get('default_setpoint')

        self.fmu_file = self.config.get('fmu_file')
        self.boiler = load_fmu(self.fmu_file)

        self.model_options = self.boiler.simulate_options()
        self.model_options['initialize'] = True
        self._model_update_rate = self.config.get('model_update_rate', 30)
        self.model_options['CVode_options']['rtol'] = 1e-6
        self.model_options['CVode_options']['atol'] = 1e-8

        self.initialize_boiler_cdl()

        # self.app = Flask('boiler_controller')
        # self.app.add_url_rule('/get_data', 'get_data', self.get_readings)
        # self.web_api_port = self.config.get('web_api_port', 5000)

        self.loop = asyncio.get_event_loop()
        self.schedule_tasks()


    def initialize_boiler_cdl(self):
        boiler_inputs = self.get_current_state()


        # boiler_inputs_SI_units = self.convert_units(boiler_inputs)

        ## TODO: Set once
        self.boiler.set('nPum', 2)
        self.boiler.set('nSta', 1)
        self.boiler.set('nBoi', 2)

        self.boiler.set('nHotWatResReqIgn', 2) # number of requests to be ignored
        self.boiler.set('TPlaHotWatSetMax', 353.15) ## also initial temperature setpoint for the trim and respond sequence
        self.boiler.set('TConBoiHotWatSetMax', 353.15)
        self.boiler.set('THotWatSetMinConBoi', 305.37)

        # uStaCha, uHotWatPumSta[nPum], nHotWatSupResReq, uTyp[nSta], uCurStaSet
        # op: TPlaHotWatSupSet

        inputs = (
                    #change input variables below
                    ['uStaCha', 'uHotWatPumSta[1]', 'uHotWatPumSta[2]', 'nHotWatSupResReq', 'uTyp[1]', 'uCurStaSet'],
                    np.array(
                        [[0, False, True, True, boiler_inputs.get('num_requests'), 1, 1],
                         [30, False, True, True, boiler_inputs.get('num_requests'), 1, 1]]
                    )
                )
        self.boiler.simulate(0, 30, inputs, options=self.model_options)
        self.model_options['initialize'] = False
        print(self.boiler.get('TPlaHotWatSupSet'))
        self.current_time = 30

    def schedule_tasks(self):
        self.loop.create_task(self._periodic_advance_time())
        self.loop.create_task(self._update_state())

    async def _update_state(self):
        while True:
            print("hello")
            await asyncio.sleep(5)

    def get_current_state(self): 
        ## get data from BMS and return a dictionary 

        return {'num_requests': 3}

    async def _periodic_advance_time(self):
        while True:
            print("current time == {}".format(self.current_time))
            start = self.current_time
            end = self.current_time + self._model_update_rate

            boiler_values = self.get_current_state()

            inputs = (
                    #change input variables below
                    ['uStaCha', 'uHotWatPumSta[1]', 'uHotWatPumSta[2]', 'nHotWatSupResReq', 'uTyp[1]', 'uCurStaSet'],
                    np.array(
                        [[start, False, True, True, boiler_values.get('num_requests'), 1, 1],
                         [end, False, True, True, boiler_values.get('num_requests'), 1, 1]]
                    )
                )
            self.boiler.simulate(start, end, inputs, options=self.model_options)

            latest_boiler_setpoint = self.boiler.get('TPlaHotWatSupSet')[0]
            print(latest_boiler_setpoint)

            # return latest_boiler_setpoint <<<---- CARLOS: use this to generate setpoint 
            ## TODO: self.set_boiler_setpoint(latest_boiler_setpoint (F)) 

            self.current_time = self.current_time + self._model_update_rate
            await asyncio.sleep(self._model_update_rate)


def main():

    try:
        loop = asyncio.get_event_loop()
        boiler = Boiler_Controller()
        # threading.Thread(target=boiler.run).start()
        loop.run_forever()

    except KeyboardInterrupt:
        print('Stopping event loop')
        loop.stop()

if __name__ == "__main__":
    main()
