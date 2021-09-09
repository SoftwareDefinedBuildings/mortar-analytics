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

    def __init__(self, boiler_name, hw_consumers, brick_model, bacpypesAPP):

        self.boiler_name    = boiler_name
        self.hw_consumers   = hw_consumers
        self.brick_model    = brick_model
        self.bacpypesAPP    = bacpypesAPP
        self.htm            = None
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
        self.check_for_htm_terminals()
        self.clean_hw_consumers()


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
        terminal_short_name = self.hw_consumers.loc[:, "equip_type"].str.split("#").str[1]
        unique_terminals = terminal_short_name.unique()
        htm = [unit in self.htm_terminals for unit in unique_terminals]
        self.hw_consumers.loc[:, "htm"] = terminal_short_name.isin(unique_terminals[htm])

        # return valves to use in determination of hot water requests
        htm_vlvs = []
        if any(htm):
            for htm_cons in unique_terminals[htm]:
                htm_vlvs.append(self._query_htm_valves(htm_cons))

        df_vlvs = pd.concat(htm_vlvs)
        self.htm =  bms.SaveBacnetPoints(df_vlvs, self.bacpypesAPP, timezone='US/Pacific', 
                                            prj_folder='./', data_file='rad_vlv_measurements'
                                            )

        # add bacnet id to self.hw_consumers
        self.hw_consumers = pd.merge(self.hw_consumers, df_vlvs.loc[:, ["rad_panel", "bacnet_instance"]], how="left", left_on="t_unit", right_on="rad_panel")


        # start collecting data
        if self._debug: print("Starting archiving htm terminal control data.\n")
        archive_htm_data = threading.Thread(target=self.collect_htm_data)
        archive_htm_data.daemon = True
        archive_htm_data.start()


    def collect_htm_data(self):
        """
        Archive high thermal mass control valve data
        """
        while True:
            self.htm.archive_sensor_values()
            time.sleep(self.htm.reading_rate)


    def clean_hw_consumers(self):
        """
        Make sure metadata only include unique terminal units
        and initiate last 2 columns for holding last request values and time
        """

        self.hw_consumers = self.hw_consumers.drop_duplicates(subset=["t_unit", "equip_type"]).reset_index(drop=True)

        # initiate containers
        self.hw_consumers.loc[:, "last_req_val"] = None
        self.hw_consumers.loc[:, "last_req_time"] = None


    def get_boiler_setpoint(self):
        """
        Retreive current boiler setpoint
        """

        return self.get_point_value(self.boiler_points["Supply_Water_Temperature_Setpoint"])

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

        # htm_req_count = self.get_htm_request()
        # htm_req_count = 0
        req_count = fr_req_count + htm_req_count

        # save to file
        with open('./DATA/number_of_request.csv', mode='a') as f:
            f.write(f"{fr_req_count},{htm_req_count},{req_count},{pd.Timestamp.now()}\n")
        f.close()

        #return req_count

    def test_num_request(self):

        try:
            while True:
                self.get_num_requests()
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
        if self._debug: print("Starting running test on determining requests.\n")
        run_test = threading.Thread(target=self.test_num_request)
        #run_test.daemon = True
        run_test.start()


    def get_fr_requests(self):
        """
        Determine the number of hot water requests from fast reacting terminal units
        """
        quick_consumers = self.hw_consumers.loc[~self.hw_consumers.loc[:, "htm"], :]

        threshold_position = 50.0

        req_count = 0
        for i, unit in quick_consumers.iterrows():
            consumer_vlv = self._query_hw_consumer_valve(unit["t_unit"])

            if consumer_vlv.shape[0] > 1:
                if "TABs_Radiant_Loop" in unit["t_unit"]:
                    # TODO Figure out how to return only the heating valve
                    consumer_vlv = consumer_vlv.loc[["RASELVLV1_2-O" in pt for pt in consumer_vlv["point_name"]], :]
                if "Heat_Pump" in unit["equip_type"]:
                    consumer_vlv = consumer_vlv.loc[["ISOVLV" in pt for pt in consumer_vlv["point_name"]], :]
            elif consumer_vlv.shape[0] == 0:
                print(f"{unit['t_unit']} does not have a control valve!\n")
                continue

            # get valve values
            vlv_val = self.get_point_value(consumer_vlv.iloc[0])
            vlv_timestamp = pd.Timestamp.now()

            if self._debug: print(vlv_val, " | ", unit["t_unit"], " | ")

            # save to container
            self.hw_consumers.loc[unit.name, "last_req_val"] = vlv_val
            self.hw_consumers.loc[unit.name, "last_req_time"] = vlv_timestamp

            if isinstance(vlv_val, str):
                if vlv_val.lower() in ['active', 'on']:
                    req_count+=1
            elif "binary" in str(consumer_vlv.iloc[0]["bacnet_type"]):
                if vlv_val == 1:
                    req_count+=1
            elif "PERCENT" in str(consumer_vlv.iloc[0]["val_unit"]) or vlv_val > 1:
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
        slow_consumers = self.hw_consumers.loc[self.hw_consumers.loc[:, "htm"], :]
        threshold_time = 6*3600
        threshold_position = 0.5

        # read in htm terminal data
        point_records = self.read_json(self.htm.data_file)
        dfs_htm = self.records_to_df(point_records)

        req_count = 0
        for key in dfs_htm.keys():
            htm_req = 0
            df = dfs_htm[key]
            time_diff = df.index.to_series().diff()
            on_instances = time_diff[df > threshold_position]

            if len(on_instances) > 1:
                on_time = on_instances.sum()

                if on_time.total_seconds() > threshold_time:
                    req_count+=1
                    htm_req = 1

            # add information to hw_consumers container
            key_id = self.hw_consumers.loc[:, "bacnet_instance"].astype(str) == key
            self.hw_consumers.loc[key_id, "last_req_val"] = htm_req
            self.hw_consumers.loc[key_id, "last_req_time"] = pd.Timestamp.now()

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


    def _query_htm_valves(self, htm_terminal):
        """
        Retrieve radiant manifold valve metadata and bacnet information
        """

        vlv_query = f"""SELECT DISTINCT * WHERE {{
            ?point_name     rdf:type                        brick:Valve_Command .
            ?point_name     brick:isPointOf                 ?rad_panel .
            ?rad_panel      rdf:type/rdfs:subClassOf?       brick:{htm_terminal} .
            ?point_name     brick:hasUnit                   ?unit .
            ?point_name     brick:bacnetPoint               ?bacnet_id .
            ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
            ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
            ?bacnet_id      brick:accessedAt                ?bacnet_net .
            ?bacnet_net     dbc:connstring                  ?bacnet_addr .
        }}
        """

        q_result = self.brick_model.query(vlv_query)
        df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

        # drop duplicate valve commands
        df = df.drop_duplicates(subset=['point_name']).reset_index(drop=True)

        return df


    def _query_hw_consumer_valve(self, hw_consumer):
        """
        Retrieve control valves for hot water consumers
        """

        vlv_query2 = """SELECT DISTINCT * WHERE {
            ?point_name     rdf:type                        brick:Position_Sensor .
            ?point_name     brick:isPointOf                 ?t_unit .
            ?point_name     brick:bacnetPoint               ?bacnet_id .
            ?point_name     brick:hasUnit?                  ?val_unit .
            ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
            ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
            ?bacnet_id      brick:accessedAt                ?bacnet_net .
            ?bacnet_net     dbc:connstring                  ?bacnet_addr .
        }"""

        vlv_query3 = """SELECT DISTINCT * WHERE {
            ?point_name     rdf:type                        brick:Valve_Command .
            ?point_name     brick:isPointOf                 ?t_unit .
            ?point_name     brick:bacnetPoint               ?bacnet_id .
            ?point_name     brick:hasUnit?                  ?val_unit .
            ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
            ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
            ?bacnet_id      brick:accessedAt                ?bacnet_net .
            ?bacnet_net     dbc:connstring                  ?bacnet_addr .
        }"""

        vlv_query4 = """SELECT DISTINCT * WHERE {
            { ?point_name rdf:type  brick:Position_Sensor } UNION { ?point_name rdf:type  brick:Valve_Command }.
            ?point_name     brick:isPointOf                 ?t_unit .
            ?point_name     brick:bacnetPoint               ?bacnet_id .
            ?point_name     brick:hasUnit?                  ?val_unit .
            ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
            ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
            ?bacnet_id      brick:accessedAt                ?bacnet_net .
            ?bacnet_net     dbc:connstring                  ?bacnet_addr .
        }"""

        vlv_query = """SELECT DISTINCT * WHERE {
            VALUES ?ctrl_equip { brick:Position_Sensor brick:Valve_Command }
            ?point_name     rdf:type                        ?ctrl_equip .
            ?point_name     brick:isPointOf                 ?t_unit .
            ?point_name     brick:bacnetPoint               ?bacnet_id .
            ?point_name     brick:hasUnit?                  ?val_unit .
            ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
            ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
            ?bacnet_id      brick:accessedAt                ?bacnet_net .
            ?bacnet_net     dbc:connstring                  ?bacnet_addr .
        }"""

        q_result = self.brick_model.query(vlv_query, initBindings={"t_unit": hw_consumer})
        df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

        df = df.drop_duplicates(subset=['point_name']).reset_index(drop=True)

        return df


    ## return: {'num_requests': int, 'max_temperature_setpoint': float (F), 'min_temperature_setpoint': float (F), 'current_setpoint': float (F)}


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
        if self._debug: print(f"Retrieving {brick_point_class} BACnet information for {equip_name}\n")
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