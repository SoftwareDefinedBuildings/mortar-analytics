#TODO: add bounds for temperature setpoints

from pyfmi import load_fmu
import yaml
import numpy as np
import asyncio
import brickschema

import bacnet_and_data.readWriteProperty as BACpypesAPP
from bacnet_and_data.ControlledBoiler import ControlledBoiler as Boiler

# from flask import Flask, jsonify
import threading

# import datetime
# import pytz
import pandas as pd

## TODO: include get_current_state(): both fast system and slower radiant system
## TODO: logging functions
## TODO: set initial setpoint to minimum setpoint --> currently in boiler CDL -> t&R block; not exposed. ask michael

# debugging setting
_debug = 1

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

        # brick model setting
        self.brick_file = self.config.get('brick_file')
        self.bacnet_ini_file = self.config.get('bacnet_init_file')

        self.load_brick_model()
        self.initialize_bacnet_comm()
        self.initialize_bldg_boilers()

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


    def load_brick_model(self):
        if _debug: print("Loading existing, pre-expanded building Brick model.\n")
        g = brickschema.Graph()
        self.brick_model = g.load_file(self.brick_file)


    def initialize_bacnet_comm(self):
        if _debug: print("\nSetting up communication with BACnet network.\n")
        self.access_bacnet = BACpypesAPP.Init(self.bacnet_ini_file)


    def query_hw_consumers(self):
        """
        Retrieve hot water consumers in the building, their respective
        boiler(s), and relevant hvac zones.
        """
        # query direct and indirect hot water consumers
        hw_consumers_query = """SELECT DISTINCT * WHERE {
        ?boiler     rdf:type/rdfs:subClassOf?   brick:Boiler .
        ?boiler     brick:feeds+                ?t_unit .
        ?t_unit     rdf:type                    ?equip_type .
        ?mid_equip  brick:feeds                 ?t_unit .
        ?t_unit     brick:feeds+                ?room_space .
        ?room_space rdf:type/rdfs:subClassOf?   brick:HVAC_Zone .

            FILTER NOT EXISTS {
                ?subtype ^a ?t_unit ;
                    (rdfs:subClassOf|^owl:equivalentClass)* ?equip_type .
                filter ( ?subtype != ?equip_type )
                }
        }
        """
        if _debug: print("Retrieving hot water consumers for each boiler.\n")

        q_result = self.brick_model.query(hw_consumers_query)
        df_hw_consumers = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

        return df_hw_consumers


    def clean_metadata(self, df_hw_consumers):
        """
        Cleans metadata dataframe to have unique hot water consumers with
        most specific classes associated to other relevant information.
        """

        unique_t_units = df_hw_consumers.loc[:, "t_unit"].unique()
        direct_consumers_bool = df_hw_consumers.loc[:, 'mid_equip'] == df_hw_consumers.loc[:, 'boiler']

        direct_consumers = df_hw_consumers.loc[direct_consumers_bool, :]
        indirect_consumers = df_hw_consumers.loc[~direct_consumers_bool, :]

        # remove any direct hot consumers listed in indirect consumers
        for unit in direct_consumers.loc[:, "t_unit"].unique():
            indir_test = indirect_consumers.loc[:, "t_unit"] == unit

            # update indirect consumers df
            indirect_consumers = indirect_consumers.loc[~indir_test, :]

        # label type of hot water consumer
        direct_consumers.loc[:, "consumer_type"] = "direct"
        indirect_consumers.loc[:, "consumer_type"] = "indirect"

        hw_consumers = pd.concat([direct_consumers, indirect_consumers])
        hw_consumers = hw_consumers.drop(columns=["subtype"]).reset_index(drop=True)

        return hw_consumers


    def initialize_bldg_boilers(self):
        if _debug: print("Identifying each boiler's hot water consumers.\n")
        hw_consumers = self.query_hw_consumers()
        self.hw_consumers = self.clean_metadata(hw_consumers)

        boiler2control = []
        unique_boilers = self.hw_consumers['boiler'].unique()
        for b, boiler in enumerate(unique_boilers):
            boiler_consumers = self.hw_consumers.loc[:, 'boiler'].isin([boiler])
            boiler_consumers = self.hw_consumers.loc[boiler_consumers, :]
            boiler2control.append(Boiler(boiler, boiler_consumers, self.brick_model, BACpypesAPP))

            if _debug: print(f"Setup for {boiler} ({b} of {len(unique_boilers)}) is finished\n")

        self.bldg_boilers = boiler2control


    def convert_degF_to_K(self, degF):
        return round((degF - 32)/1.8 + 273.15, 2)


    def convert_K_to_degF(self, K):
        return round((K - 273.15)*1.8 + 32, 2)


    def save_new_setpoint_file(self, new_boiler_setpoint):
        # save to file
        with open('./DATA/boiler_setpoint.csv', mode='a') as f:
            f.write(f"{pd.Timestamp.now()},{new_boiler_setpoint}\n")
        f.close()


    def send_new_setpoint_to_boiler(self, new_boiler_setpoint):
        pass


    def schedule_tasks(self):
        self.loop.create_task(self._periodic_advance_time())
        self.loop.create_task(self._update_state())

    async def _update_state(self):
        while True:
            print("hello")
            await asyncio.sleep(60)

    def get_current_state(self): 
        ## get data from BMS and return a dictionary 
        num_requests = self.bldg_boilers[0].get_num_requests()
        cur_boiler_setpoint = self.bldg_boilers[0].get_boiler_setpoint()
        hw_sup_temp = self.bldg_boilers[0].get_hw_supply_temp()
        hw_ret_temp = self.bldg_boilers[0].get_hw_return_temp()
        boiler_max_sp = self.bldg_boilers[0].get_max_temp_setpoint()
        boiler_min_sp = self.bldg_boilers[0].get_min_temp_setpoint()

        current_state = {
            'num_requests': num_requests,
            'current_boiler_setpoint': self.convert_degF_to_K(cur_boiler_setpoint),
            'hw_supply_temperature': self.convert_degF_to_K(hw_sup_temp),
            'hw_return_temperature': self.convert_degF_to_K(hw_ret_temp),
            'boiler_max_setpoint': self.convert_degF_to_K(boiler_max_sp),
            'boiler_min_setpoint': self.convert_degF_to_K(boiler_min_sp)
            }

        return current_state

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
            self.save_new_setpoint_file(latest_boiler_setpoint)


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
        boiler.bldg_boilers[0].stop_htm_data_archive()
        loop.stop()

if __name__ == "__main__":
    main()
