import pandas as pd
import threading
import time
import json

import bacnet_and_data.SaveBacnetPoints as bms



class ControlledBoiler(object):
    """
    Defines a boiler in a building to implement a temperature setpoint reset
    control strategy using trim and response logic.
    """

    # debugging setting
    _debug = 1
    _verbose = 0

    def __init__(self, boiler_name, hw_consumers, brick_model, bacpypesAPP):

        self.boiler_name    = boiler_name
        self.hw_consumers   = hw_consumers
        self.brick_model    = brick_model
        self.bacpypesAPP    = bacpypesAPP
        self.htm            = None
        self.htm_archive_loop = True
        self.htm_terminals  = [
            "TABS_Panel", "ESS_Panel",
            "Thermally_Activated_Building_System_Panel",
            "Embedded_Surface_System_Panel",
            ]

        self.boiler_points  = {
            "Hot_Water_Supply_Temperature_Sensor": None,
            "Return_Water_Temperature_Sensor": None,
            "Supply_Water_Temperature_Setpoint": None,
            "High_Setpoint_Limit": None,
            "Low_Setpoint_Limit": None,
            "Enable_Command": None,
            }

        self.ctrl_params    = {
            "current_request": None,
            "htm_request": None,
            "htm_setpoint": None,
            "max_temp_setpoint": None,
            "min_temp_setpoint": None,
            }


        self.get_boiler_points()
        self.get_boiler_limits()
        self.process_hw_consumers()
        self.check_for_htm_terminals()


    def get_boiler_points(self):
        """
        Retreive bacnet information for required boiler points
        """

        for point in self.boiler_points:
            bacnet_info = self.return_equipment_points(point, self.boiler_name)

            if bacnet_info.shape[0] > 1:
                print(f"There is more than one {point} found for {self.boiler_name}\n!")
                print(f"Define which one to use manually.\n")
                print(bacnet_info.loc[:, "t_unit_point"].unique())
                # TODO: Figure out how to only get boiler temp setpoint limits
                self.boiler_points[point] = bacnet_info.loc[["SPInput" in pt for pt in bacnet_info.loc[:, "t_unit_point"]], :].iloc[0,:]
            elif bacnet_info.shape[0] == 0:
                print(f"No {point} for {self.boiler_name} found in this building!\n")
            else:
                self.boiler_points[point] = bacnet_info.iloc[0,:]


    def get_boiler_limits(self):
        """
        Retreive boiler setpoint limit values
        """

        self.ctrl_params["max_temp_setpoint"] = self.get_point_value(self.boiler_points["High_Setpoint_Limit"])
        self.ctrl_params["min_temp_setpoint"] = self.get_point_value(self.boiler_points["Low_Setpoint_Limit"])

        # test that max temperature limit is greater than minimum temperature
        if not self.ctrl_params["max_temp_setpoint"] > self.ctrl_params["min_temp_setpoint"]:
            print("Maximum boiler temperature setpoint limit is not greater than minimum limit.\n")
            import pdb; pdb.set_trace()


    def check_for_htm_terminals(self):
        """
        Check if there are any high thermal mass terminal units
        """

        # identify high thermal mass terminal units
        self.hw_consumers.loc[:, "htm"] = self.hw_consumers["equip_type"].str.contains('|'.join(self.htm_terminals))
        self.hw_consumers.loc[self.hw_consumers["htm"].isin([None]), "htm"] = False
        self.hw_consumers.loc[:, "htm"] = self.hw_consumers.loc[:, "htm"].astype('bool')

        df_vlvs = pd.DataFrame.from_records(list(self.hw_consumers.loc[self.hw_consumers["htm"], "ctrl_vlv"]))

        self.htm =  bms.SaveBacnetPoints(df_vlvs, self.bacpypesAPP, timezone='US/Pacific', 
                                            prj_folder='./', data_file='rad_vlv_measurements'
                                            )

        # start collecting data
        if self._debug: print(f"[{pd.Timestamp.now()}] Starting archiving htm terminal control data.\n")
        archive_htm_data = threading.Thread(target=self.collect_htm_data)
        archive_htm_data.daemon = True
        archive_htm_data.start()


    def collect_htm_data(self):
        """
        Archive high thermal mass control valve data
        """
        while self.htm_archive_loop:
            self.htm.archive_sensor_values()
            time.sleep(self.htm.reading_rate)


    def stop_htm_data_archive(self):
        self.htm_archive_loop = False


    def process_hw_consumers(self):
        """
        Process hot water consumers to their type, controlling valve bacnet info,
        mode status bacnet info, and verify they are unique.
        Also, initiate last 2 columns for holding last request values and time
        """

        # return hot water consumers most specific entity class
        terminal_units = self.hw_consumers["t_unit"]
        entity_class = [self.return_class_entity(t_unit) for t_unit in terminal_units]
        self.hw_consumers.loc[:, "equip_type"] = entity_class

        # return hot water consumers controlling valve
        control_vlvs = [self.query_hw_consumer_valve(t_unit) for t_unit in terminal_units]
        self.hw_consumers.loc[:, "ctrl_vlv"] = control_vlvs

        # return hot water consumers mode status
        mode_status = [self.query_hw_mode_status(t_unit) for t_unit in terminal_units]
        self.hw_consumers.loc[:, "mode_status"] = [m[0] for m in mode_status]
        self.hw_consumers.loc[:, "mode_status_type"] = [m[1] for m in mode_status]

        # verify that hot water consumers are unique
        self.hw_consumers = self.hw_consumers.drop_duplicates(subset=["t_unit", "equip_type"]).reset_index(drop=True)

        # initiate containers
        self.hw_consumers.loc[:, "last_req_val"] = None
        self.hw_consumers.loc[:, "last_req_time"] = None
        self.hw_consumers.loc[:, "last_req_mode"] = None


    def get_boiler_setpoint(self):
        """
        Retreive current boiler setpoint
        """

        return self.get_point_value(self.boiler_points["Supply_Water_Temperature_Setpoint"])


    def get_boiler_status(self):
        """
        Retreive boiler maximum temperature setpoint
        """

        return self.get_point_value(self.boiler_points["Enable_Command"])


    def get_hw_supply_temp(self):
        """
        Retreive current hot water supply temperature
        """

        return self.get_point_value(self.boiler_points["Hot_Water_Supply_Temperature_Sensor"])


    def get_hw_return_temp(self):
        """
        Retreive current hot water return temperature
        """

        return self.get_point_value(self.boiler_points["Return_Water_Temperature_Sensor"])


    def get_max_temp_setpoint(self):
        """
        Retreive boiler maximum temperature setpoint
        """

        return self.get_point_value(self.boiler_points["High_Setpoint_Limit"])


    def get_min_temp_setpoint(self):
        """
        Retreive boiler maximum temperature setpoint
        """

        return self.get_point_value(self.boiler_points["Low_Setpoint_Limit"])


    def get_num_requests(self):
        """
        Determine total number of request for Boiler
        """

        fr_req_count = self.get_fr_requests()

        htm_req_count = self.get_htm_request()

        req_count = fr_req_count

        # save to file
        with open('./DATA/number_of_request.csv', mode='a') as f:
            f.write(f"{fr_req_count},{htm_req_count},{req_count},{pd.Timestamp.now()}\n")
        f.close()

        return req_count


    def write_new_boiler_setpoint(self, new_value, priority=13):

        self.write_point_value(self.boiler_points["Supply_Water_Temperature_Setpoint"], new_value, priority)


    def test_num_request(self):

        try:
            while True:
                req_count = self.get_num_requests()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt has been caught.")
            print("\nGoing into debug mode...")
            print("\nYou can enter 'c' to continue after finished making modifications or debugging..")
            print("Or enter quit() to quit python script.")
            import pdb; pdb.set_trace()

            self.test_num_request()


    def run_test(self):
        # start request calcs test
        if self._debug: print(f"[{pd.Timestamp.now()}] Starting running test on determining requests.\n")
        run_test = threading.Thread(target=self.test_num_request)
        #run_test.daemon = True
        run_test.start()


    def get_fr_requests(self):
        """
        Determine the number of hot water requests from fast reacting terminal units
        """
        if self._debug: print(f"[{pd.Timestamp.now()}] Starting to determine hotter water requests for fast reacting units.\n")

        quick_consumers = self.hw_consumers.loc[~self.hw_consumers.loc[:, "htm"], :]
        threshold_position = 95.0

        req_count = 0
        for i, unit in quick_consumers.iterrows():

            # get valve status
            consumer_vlv = unit["ctrl_vlv"]
            if consumer_vlv is None:
                print(f"{unit['t_unit']} does not have a control valve!\n")
                continue

            vlv_val = self.get_point_value(consumer_vlv)
            vlv_timestamp = pd.Timestamp.now()

            # get mode status if available
            consumer_heat_mode = False
            if unit["mode_status"] is None:
                consumer_heat_mode = True
            else:
                cur_mode_status = self.get_point_value(unit["mode_status"])

                if unit["mode_status_type"] == "heating":
                    if cur_mode_status > 0:
                        consumer_heat_mode = True

            if self._debug and self._verbose: print(vlv_val, " | ", unit["t_unit"], " | ")

            # save to container
            self.hw_consumers.loc[unit.name, "last_req_val"] = vlv_val
            self.hw_consumers.loc[unit.name, "last_req_time"] = vlv_timestamp
            self.hw_consumers.loc[unit.name, "last_req_mode"] = consumer_heat_mode

            if consumer_heat_mode:
                if isinstance(vlv_val, str):
                    if vlv_val.lower() in ['active', 'on']:
                        req_count+=1
                elif "binary" in str(consumer_vlv["bacnet_type"]):
                    if vlv_val == 1:
                        req_count+=1
                elif "PERCENT" in str(consumer_vlv["val_unit"]) or vlv_val > 1:
                    if vlv_val > threshold_position:
                        req_count+=1
                else:
                    if vlv_val > threshold_position/100:
                        req_count+=1

        return req_count


    def get_htm_request(self):
        """
        Calcuate the number of requests for high thermal mass terminal units
        """
        if self._debug: print(f"[{pd.Timestamp.now()}] Starting to determine hotter water requests for slow reacting units.\n")
        slow_consumers = self.hw_consumers.loc[self.hw_consumers.loc[:, "htm"], :]
        threshold_time = 6*3600
        threshold_position = 0.5

        # read in htm terminal data
        point_records = self.read_json(self.htm.data_file)
        dfs_htm = self.records_to_df(point_records)

        req_count = 0
        for i, unit in slow_consumers.iterrows():
            htm_req = 0
            key = str(unit['ctrl_vlv']['bacnet_instance'])

            df = dfs_htm[key]
            time_diff = df.index.to_series().diff()
            on_instances = time_diff[df > threshold_position]

            if len(on_instances) > 1:
                on_time = on_instances.sum()

                if on_time.total_seconds() > threshold_time:
                    req_count+=1
                    htm_req = 1

            # add information to hw_consumers container
            self.hw_consumers.loc[unit.name, "last_req_val"] = htm_req
            self.hw_consumers.loc[unit.name, "last_req_time"] = pd.Timestamp.now()

        return req_count


    def get_point_value(self, bacnet_info):
        """
        Get current value from sensor in a BACnet network
        """
        read_attr = "presentValue"

        obj_type = bacnet_info['bacnet_type']
        obj_inst = bacnet_info['bacnet_instance']
        bacnet_addr = bacnet_info['bacnet_addr']

        bacnet_args = [bacnet_addr, obj_type, obj_inst, read_attr]

        val = self.bacpypesAPP.read_prop(bacnet_args)

        if val is None:
            val = self.bacpypesAPP.read_prop(bacnet_args)

        if "binaryOutput" in str(obj_type):
            val = self.convert_to_int(val)

        return val


    def write_point_value(self, bacnet_info, new_value, priority=13):
        """
        Write new value to bacnet point
        """

        write_attr = "presentValue"

        obj_type = bacnet_info['bacnet_type']
        obj_inst = bacnet_info['bacnet_instance']
        bacnet_addr = bacnet_info['bacnet_addr']

        bacnet_args = [bacnet_addr, obj_type, obj_inst, write_attr, str(new_value), '-', priority]

        self.bacpypesAPP.write_prop(bacnet_args)


    def query_hw_consumer_valve(self, hw_consumer):
        """
        Retrieve control valves for hot water consumers
        """

        vlv_query = """SELECT DISTINCT * WHERE {
            VALUES ?ctrl_equip { brick:Position_Sensor brick:Valve_Command }
            ?point_name     rdf:type                        ?ctrl_equip .
            ?point_name     brick:isPointOf                 ?t_unit .
            ?point_name     brick:bacnetPoint               ?bacnet_id .
            ?point_name     brick:hasUnit?                  ?val_unit .
            ?val_unit       rdf:type                        ?units_type .
            ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
            ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
            ?bacnet_id      brick:accessedAt                ?bacnet_net .
            ?bacnet_net     dbc:connstring                  ?bacnet_addr .

            FILTER NOT EXISTS {
                VALUES ?exclude_tags {tag:Reversing tag:Damper }
                ?point_name brick:hasTag ?exclude_tags.
            }
        }"""

        q_result = self.brick_model.query(vlv_query, initBindings={"t_unit": hw_consumer})
        df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

        # Verify that unit is a qudt unit
        #TODO figure out why brick:hasUnit is returning other objects
        qudt_units = df["val_unit"].str.contains("qudt.org")
        if any(qudt_units):
            df = df.loc[qudt_units, :]
        else:
            df.loc[~qudt_units, "val_unit"] = None

        df = df.drop_duplicates(subset=['point_name']).reset_index(drop=True)

        if df.shape[0] == 0:
            return None

        return dict(df.loc[df.index[0]])

    def query_hw_mode_status(self, hw_consumer):
        """
        Get mode status (heating or cooling) of hot water consumers
        """

        mode_query = """SELECT DISTINCT * WHERE {
            VALUES ?mode_status {
                brick:Heating_Start_Stop_Status brick:Heating_Enable_Command
                brick:Cooling_Start_Stop_Status brick:Cooling_Enable_Command
                }

            ?point_name     rdf:type                        ?mode_status .
            ?point_name     brick:isPointOf                 ?t_unit .
            ?point_name     brick:bacnetPoint               ?bacnet_id .
            ?point_name     brick:hasUnit?                  ?val_unit .
            ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
            ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
            ?bacnet_id      brick:accessedAt                ?bacnet_net .
            ?bacnet_net     dbc:connstring                  ?bacnet_addr .
        }"""

        q_result = self.brick_model.query(mode_query, initBindings={"t_unit": hw_consumer})
        df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

        # Verify that unit is a qudt unit
        #TODO figure out why brick:hasUnit is returning other objects
        qudt_units = df["val_unit"].str.contains("qudt.org")
        if any(qudt_units):
            df = df.loc[qudt_units, :]
        else:
            df.loc[~qudt_units, "val_unit"] = None

        status_entities = {
            "Heating_Start_Stop_Status": True,
            "Cooling_Start_Stop_Status": False,
            "Heating_Enable_Command": True,
            "Cooling_Enable_Command": False,
        }

        indicate_status = None
        if df.shape[0] > 1:
            for entity in status_entities:
                mode_found = df["mode_status"].str.contains(entity)
                if any(mode_found):
                    df = df.loc[mode_found, :]
                    if status_entities[entity]:
                        indicate_status = 'heating'
                    else:
                        indicate_status = 'not heating'
        else:
            return (None, None)

        # for col in df.columns: print(col, df[col].unique(), '\n')

        return (dict(df.loc[df.index[0]]), indicate_status)


    def return_class_entity(self, entity):
        """
        Return the most specific class for an entity
        """

        term_query = """ SELECT * WHERE {
            ?entity rdf:type/rdfs:subClassOf? ?entity_class

                FILTER NOT EXISTS {
                    ?subtype ^a ?entity ;
                        (rdfs:subClassOf|^owl:equivalentClass)* ?entity_class .
                    filter ( ?subtype != ?entity_class )
                    }
        }
        """

        term_query_result = self.brick_model.query(term_query, initBindings={"entity": entity})
        df_term_query_result = pd.DataFrame(term_query_result, columns=[str(s) for s in term_query_result.vars])

        entity_class = df_term_query_result["entity_class"].unique()

        if len(entity_class) > 1:
            entity_class = self.brick_model.get_most_specific_class(entity_class)

        if len(entity_class) == 0:
            return None

        return entity_class[0]


    def return_equipment_points(self, brick_point_class, equip_name):
        """
        Return defined brick point class for piece of equipment
        """
        # query to return all setpoints of equipment
        term_query = f"""SELECT DISTINCT * WHERE {{
            ?t_unit         brick:hasPoint                  ?t_unit_point .
            ?t_unit_point   rdf:type/rdfs:subClassOf*       brick:{brick_point_class} .
            ?t_unit_point   rdf:type                        ?point_type .

            ?t_unit_point   brick:bacnetPoint               ?bacnet_id .
            ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
            ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
            ?bacnet_id      brick:accessedAt                ?bacnet_net .
            ?bacnet_net     dbc:connstring                  ?bacnet_addr .

            FILTER NOT EXISTS {{
                ?subtype ^a ?t_unit_point ;
                    (rdfs:subClassOf|^owl:equivalentClass)* ?point_type .
                filter ( ?subtype != ?point_type )
                }}
            }}"""

        # execute the query
        if self._debug and self._verbose: print(f"[{pd.Timestamp.now()}] Retrieving {brick_point_class} BACnet information for {equip_name}\n")
        term_query_result = self.brick_model.query(term_query, initBindings={"t_unit": equip_name})

        df_term_query_result = pd.DataFrame(term_query_result, columns=[str(s) for s in term_query_result.vars])
        df_term_query_result.loc[:, "short_point_type"] = [shpt.split("#")[1] for shpt in df_term_query_result["point_type"]]
        df_unique_stpt_class = df_term_query_result.drop_duplicates(subset=["t_unit_point", "point_type"])

        return df_unique_stpt_class


    def convert_to_int(self, point_value):

        if isinstance(point_value, str):
            if point_value.lower() in ['active', 'on']:
                point_value = 1.0
            elif point_value.lower() in ['inactive', 'off']:
                point_value = 0.0

        return point_value


    def read_json(self, path):
        """
        Read measurement data contained in json format
        """

        with open(path, mode="r") as f:
            data = json.load(f)
        f.close()

        return data


    def records_to_df(self, point_records):
        """
        Convert json sensor records to pandas dataframe
        """
        dfs = {}

        for sensor_id in point_records.keys():
            idx = []
            val = []
            sensor_name = point_records[sensor_id]["name"]
            for reading_time, sensor_value in point_records[sensor_id]["readings"]:
                readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reading_time))
                idx.append(readable_time)
                val.append(sensor_value)

                dat = pd.Series(data=val, index=idx, name=sensor_name)
                dat.index = pd.to_datetime(dat.index)

            dfs[sensor_id] = dat

        return dfs