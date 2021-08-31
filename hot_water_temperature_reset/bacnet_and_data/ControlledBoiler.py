import pandas as pd


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

        import pdb; pdb.set_trace()


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
                import pdb; pdb.set_trace()
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

        unique_terminals = self.hw_consumers.loc[:, "equip_type"].str.split("#").str[1].unique()

        htm = [unit in self.htm_terminals for unit in unique_terminals]

        htm_vlvs = []
        if any(htm):
            for htm_cons in unique_terminals[htm]:
                htm_vlvs.append(self._query_htm_valves(htm_cons))

        df_vlvs = pd.concat(htm_vlvs)
        import pdb; pdb.set_trace()
        self.htm =  bms.SaveBacnetPoints(df_vlvs, self.bacpypesAPP, timezone='US/Pacific', 
                                            prj_folder='./', data_file='rad_vlv_measurements'
                                            )



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
        Calculate the number of request for the boiler
        """
        pass


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