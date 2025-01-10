#TODO: add bounds for temperature setpoints
from pyfmi import load_fmu
import yaml
import numpy as np
import asyncio
import brickschema
import logging, sys
from datetime import date

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

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('brick_enabled_cdl_controller.log')

logger.addHandler(c_handler)
logger.addHandler(f_handler)


# debugging setting
_debug = 1

class Boiler_Controller:
    def __init__(self, config_file='config.yaml'):
        with open(config_file) as fp:
            self.config = yaml.safe_load(fp)

        self._default_setpoint = self.config.get('default_setpoint')

        self.prev_calc_boiler_setpoint = None
        self.fmu_file = self.config.get('fmu_file')
        self._model_update_rate = self.config.get('model_update_rate', 300)

        # randomized control experiment setup
        self.randomized_ctrl_sch_file = self.config.get('randomized_ctrl_sch', None)
        self.default_ctrl_strategy = self.config.get('default_ctrl_strategy', 'g36')
        self.randomized_ctrl_sch = None
        self.processed_ctrl_sch = False
        self.initialize_ctrl_sch()

        # brick model setting
        self.brick_file = self.config.get('brick_file')
        self.bacnet_ini_file = self.config.get('bacnet_init_file')

        self.sim_boiler_status = True
        self.switch_boiler_time = pd.Timestamp.now() + pd.Timedelta('1hour')
        self.load_brick_model()
        self.initialize_bacnet_comm()
        self.initialize_bldg_boilers()
        self.prev_ctrl_type = None

        # self.app = Flask('boiler_controller')
        # self.app.add_url_rule('/get_data', 'get_data', self.get_readings)
        # self.web_api_port = self.config.get('web_api_port', 5000)

        self.loop = asyncio.get_event_loop()
        self.schedule_tasks()

    def initialize_ctrl_sch(self):
        # process ctrl sch
        if self.randomized_ctrl_sch_file is not None:
            self.randomized_ctrl_sch = pd.read_csv(self.randomized_ctrl_sch_file)
            self.randomized_ctrl_sch['strategy'].loc[self.randomized_ctrl_sch['strategy'].isin([1])] = 'baseline'
            self.randomized_ctrl_sch['strategy'].loc[self.randomized_ctrl_sch['strategy'].isin([2])] = 'g36'

            # make dates the index for quick selections
            self.randomized_ctrl_sch = self.randomized_ctrl_sch.set_index('date')
            self.processed_ctrl_sch = True
            ctrl_strategies = np.unique(self.randomized_ctrl_sch['strategy'])
            if _debug: logger.info(f"[{pd.Timestamp.now()}] Initialized randomized control schedule with strategies {ctrl_strategies}\n")


    def initialize_boiler_cdl(self):
        self.boiler = load_fmu(self.fmu_file)
        self.model_options = self.boiler.simulate_options()
        self.model_options['initialize'] = True
        self.model_options['CVode_options']['rtol'] = 1e-6
        self.model_options['CVode_options']['atol'] = 1e-8

        boiler_inputs = self.get_current_state()

        # boiler_inputs_SI_units = self.convert_units(boiler_inputs)

        ## TODO: Set once
        self.boiler.set('nPum', 2)
        self.boiler.set('nSta', 1)
        self.boiler.set('nBoi', 2)

        self.boiler.set('nHotWatResReqIgn', 2) # number of requests to be ignored
        self.boiler.set('TPlaHotWatIniSet', boiler_inputs.get('current_boiler_setpoint', self.convert_degF_to_K(130))) ## initial temperature setpoint
        self.boiler.set('TPlaHotWatSetMax', self.convert_degF_to_K(130))
        self.boiler.set('TConBoiHotWatSetMax', self.convert_degF_to_K(130))
        self.boiler.set('THotWatSetMinConBoi', self.convert_degF_to_K(80))

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
        latest_boiler_setpoint = round(self.boiler.get('TPlaHotWatSupSet')[0], 2)
        self.prev_calc_boiler_setpoint = latest_boiler_setpoint
        logger.info(f"[{pd.Timestamp.now()}] Initialized boiler setpoint {latest_boiler_setpoint} K ({self.convert_K_to_degF(latest_boiler_setpoint)} degF)")
        self.cdl_sim_time = 30


    def load_brick_model(self):
        if _debug: logger.info(f"[{pd.Timestamp.now()}] Loading existing, pre-expanded building Brick model.\n")
        g = brickschema.Graph()
        self.brick_model = g.load_file(self.brick_file)


    def initialize_bacnet_comm(self):
        if _debug: logger.info(f"[{pd.Timestamp.now()}] Setting up communication with BACnet network.\n")
        self.access_bacnet = BACpypesAPP.Init(self.bacnet_ini_file)


    def query_hw_consumers(self):
        """
        Retrieve hot water consumers in the building, their respective
        boiler(s), and relevant hvac zones.
        """
        # query direct and indirect hot water consumers
        hw_consumers_query = """ SELECT DISTINCT * WHERE {
            VALUES ?t_type { brick:Equipment brick:Water_Loop }
            ?boiler     rdf:type/rdfs:subClassOf?   brick:Boiler .
            ?boiler     brick:feeds+                ?t_unit .
            ?t_unit     brick:isFedBy               ?mid_equip .
            ?t_unit     rdf:type/rdfs:subClassOf?   ?t_type .
        }
        """
        if _debug: logger.info(f"[{pd.Timestamp.now()}] Retrieving hot water consumers for each boiler.\n")

        q_result = self.brick_model.query(hw_consumers_query)
        df_hw_consumers = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

        # Verify that intermediate equipment is part of hot water system
        boilers = list(df_hw_consumers["boiler"].unique())
        terminal_units = list(df_hw_consumers["t_unit"].unique())
        part_hw_sys = df_hw_consumers["mid_equip"].isin(boilers + terminal_units)

        return df_hw_consumers.loc[part_hw_sys, :]


    def clean_metadata(self, df_hw_consumers):
        """
        Cleans metadata dataframe to have unique hot water consumers with
        most specific classes associated to other relevant information.
        """

        direct_consumers_bool = df_hw_consumers.loc[:, 'mid_equip'] == df_hw_consumers.loc[:, 'boiler']

        df_hw_consumers.loc[direct_consumers_bool, "consumer_type"] = "direct"
        df_hw_consumers.loc[~direct_consumers_bool, "consumer_type"] = "indirect"

        return df_hw_consumers


    def initialize_bldg_boilers(self):
        if _debug: logger.info(f"[{pd.Timestamp.now()}] Identifying each boiler's hot water consumers.\n")
        hw_consumers = self.query_hw_consumers()
        self.hw_consumers = self.clean_metadata(hw_consumers)

        boiler2control = []
        unique_boilers = self.hw_consumers['boiler'].unique()
        for b, boiler in enumerate(unique_boilers):
            boiler_consumers = self.hw_consumers.loc[:, 'boiler'].isin([boiler])
            boiler_consumers = self.hw_consumers.loc[boiler_consumers, :]
            boiler2control.append(Boiler(boiler, boiler_consumers, self.brick_model, BACpypesAPP))

            if _debug: logger.info(f"[{pd.Timestamp.now()}] Setup for {boiler} ({b+1} of {len(unique_boilers)}) is finished\n")

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


    def save_sim_boiler_status(self, new_boiler_status):
        # save to file
        with open('./DATA/boiler_status.csv', mode='a') as f:
            f.write(f"{pd.Timestamp.now()},{new_boiler_status}\n")
        f.close()


    def send_new_setpoint_to_boiler(self, new_boiler_setpoint, priority, direction):
        degF_boiler_setpoint = round(self.convert_K_to_degF(new_boiler_setpoint), 1)

        # check that new boiler setpoint is within limits
        boiler_max_sp = self.bldg_boilers[0].get_max_temp_setpoint()
        boiler_min_sp = self.bldg_boilers[0].get_min_temp_setpoint()

        if degF_boiler_setpoint > boiler_max_sp:
            degF_boiler_setpoint = boiler_max_sp
        elif degF_boiler_setpoint < boiler_min_sp:
            degF_boiler_setpoint = boiler_min_sp

        self.bldg_boilers[0].write_new_boiler_setpoint(degF_boiler_setpoint, priority, direction)


    def schedule_tasks(self):
        self.loop.create_task(self._periodic_advance_time())
        self.loop.create_task(self._update_state())

    async def _update_state(self):
        while True:
            logger.info(f"[{pd.Timestamp.now()}] Controller tick ...")
            await asyncio.sleep(60)

    def get_current_state(self): 
        ## get data from BMS and return a dictionary 
        boiler_status = self.bldg_boilers[0].get_boiler_status()
        num_requests = self.bldg_boilers[0].get_num_requests()
        cur_boiler_setpoint = self.bldg_boilers[0].get_boiler_setpoint()
        hw_sup_temp = self.bldg_boilers[0].get_hw_supply_temp()
        hw_ret_temp = self.bldg_boilers[0].get_hw_return_temp()
        boiler_max_sp = self.bldg_boilers[0].get_max_temp_setpoint()
        boiler_min_sp = self.bldg_boilers[0].get_min_temp_setpoint()

        current_state = {
            'boiler_status': boiler_status,
            'num_requests': num_requests,
            'current_boiler_setpoint': self.convert_degF_to_K(cur_boiler_setpoint),
            'hw_supply_temperature': self.convert_degF_to_K(hw_sup_temp),
            'hw_return_temperature': self.convert_degF_to_K(hw_ret_temp),
            'boiler_max_setpoint': self.convert_degF_to_K(boiler_max_sp),
            'boiler_min_setpoint': self.convert_degF_to_K(boiler_min_sp)
            }

        return current_state

    def simulate_boiler_status(self, current_status):
        """
        Place holder to simulate boiler on status
        """

        # prob_transition = random.uniform(0,1)

        # if current_status == True:
        #     new_status = prob_transition < .90
        # else:
        #     new_status = prob_transition < .30

        start_boiler_time = 0
        end_boiler_time = 24

        cur_time = pd.Timestamp.now()
        sch_time = cur_time.hour >= start_boiler_time and cur_time.hour < end_boiler_time

        new_status = self.sim_boiler_status
        if sch_time:
            if cur_time > self.switch_boiler_time:
                if self.sim_boiler_status:
                    self.switch_boiler_time = cur_time + pd.Timedelta('1hour')
                else:
                    self.switch_boiler_time = cur_time + pd.Timedelta('30min')
                new_status = not self.sim_boiler_status
        else:
            new_status = False


        # save enable
        logger.info(f"[{pd.Timestamp.now()}] Current Simulated Boiler Status ==  {new_status}")
        logger.info(f"[{pd.Timestamp.now()}] Switching to {not new_status} at {self.switch_boiler_time}")
        self.sim_boiler_status = new_status
        self.save_sim_boiler_status(new_status)

        return new_status

    def run_cdl_temperature_reset(self, boiler_values):
        logger.info(f"[{pd.Timestamp.now()}] CDL simulation time == {self.cdl_sim_time}")
        start = self.cdl_sim_time
        end = self.cdl_sim_time + self._model_update_rate
        pumps_enabled = boiler_values.get('boiler_status')
        num_requests = boiler_values.get('num_requests')
        inputs = (
                #change input variables below
                ['uStaCha', 'uHotWatPumSta[1]', 'uHotWatPumSta[2]', 'nHotWatSupResReq', 'uTyp[1]', 'uCurStaSet'],
                np.array(
                    [[start, False, pumps_enabled, pumps_enabled, num_requests, 1, 1],
                    [end, False, pumps_enabled, pumps_enabled, num_requests, 1, 1]]
                )
            )
        self.boiler.simulate(start, end, inputs, options=self.model_options)
        self.cdl_sim_time = self.cdl_sim_time + self._model_update_rate
        boiler_setpoint = round(self.boiler.get('TPlaHotWatSupSet')[0], 2)

        return boiler_setpoint


    def get_control_type(self):
        control_type = self.default_ctrl_strategy
        if self.processed_ctrl_sch:
            # get control type based on day
            cur_date = date.today().strftime("%Y-%m-%d")
            control_type = self.randomized_ctrl_sch.loc[cur_date, 'strategy']
        else:
            # initialize randomized schedule
            self.initialize_ctrl_sch()

        # initialize G36 controls if switch from baseline
        if control_type == 'g36':
            if self.prev_ctrl_type is None or self.prev_ctrl_type != control_type:
                self.initialize_boiler_cdl()

        self.prev_ctrl_type = control_type
        return control_type


    def get_baseline_control_sp(self):
        # return constant setpoint for baseline control
        return self.convert_degF_to_K(130) # must be in Kelvin


    async def _periodic_advance_time(self):
        while True:
            logger.info(f"[Current time is == {pd.Timestamp.now()}]")
            boiler_values = self.get_current_state()
            current_boiler_sp = boiler_values.get('current_boiler_setpoint')
            control_type = self.get_control_type()

            #TODO: When closing the loop, simulated boiler status to boiler_values.get('boiler_status')

            # pumps_enabled = self.simulate_boiler_status(self.sim_boiler_status)
            pumps_enabled = boiler_values.get('boiler_status')
            if pumps_enabled:
                if control_type == 'g36':
                    latest_boiler_setpoint = self.run_cdl_temperature_reset(boiler_values)
                elif control_type == 'baseline':
                    latest_boiler_setpoint = self.get_baseline_control_sp()

                self.prev_calc_boiler_setpoint = latest_boiler_setpoint
            else:
                logger.info(f"[{pd.Timestamp.now()}] Boiler is disabled will use last setpoint= {self.prev_calc_boiler_setpoint}")
                latest_boiler_setpoint = self.prev_calc_boiler_setpoint

            logger.info(f"[{pd.Timestamp.now()}] Current Boiler Status = {pumps_enabled}")
            logger.info(f"[{pd.Timestamp.now()}] new hot water setpoint {latest_boiler_setpoint} K ({self.convert_K_to_degF(latest_boiler_setpoint)} degF)")
            self.save_new_setpoint_file(latest_boiler_setpoint)

            # return latest_boiler_setpoint <<<---- CARLOS: use this to generate setpoint 
            ## TODO: self.set_boiler_setpoint(latest_boiler_setpoint (F))
            # Send new setpoint if different
            if abs(round(latest_boiler_setpoint, 1) - round(current_boiler_sp, 1)) > 0.1:
                if latest_boiler_setpoint > current_boiler_sp:
                    direction = 'up'
                else:
                    direction = 'down'
                logger.info(f"[{pd.Timestamp.now()}] Sending new setpoint to boiler ({self.convert_K_to_degF(latest_boiler_setpoint)})deg F at priority 13 and direction= {direction}")
                self.send_new_setpoint_to_boiler(latest_boiler_setpoint, priority=13, direction=direction)
            else:
                logger.info(f"[{pd.Timestamp.now()}] New calculate setpoint is not different enough to send to boilers. \
                    Calculated = ({self.convert_K_to_degF(latest_boiler_setpoint)})deg F | Existing = {self.convert_K_to_degF(current_boiler_sp)}deg F")


            # Verify that new setpoint was sent
            boiler_values = self.get_current_state()
            current_boiler_sp = boiler_values.get('current_boiler_setpoint')
            if abs(round(latest_boiler_setpoint, 1) - round(current_boiler_sp, 1)) > 0.3:
                logger.info(f"[{pd.Timestamp.now()}] Sent boiler setpoint DOES NOT MATCH calculated setpoint!!! PLEASE CHECK")
                logger.info(f"[{pd.Timestamp.now()}] Calculated boiler setpoint = {latest_boiler_setpoint}")
                logger.info(f"[{pd.Timestamp.now()}] Current boiler setpoint read from Boiler= {current_boiler_sp}")

            await asyncio.sleep(self._model_update_rate)


def main():

    try:
        boiler = Boiler_Controller()
        # threading.Thread(target=boiler.run).start()
        boiler.loop.run_forever()

    except KeyboardInterrupt:
        logger.info(f"[{pd.Timestamp.now()}] Stopping event loop")
        boiler.bldg_boilers[0].stop_htm_data_archive()
        boiler.loop.stop()

if __name__ == "__main__":
    main()
