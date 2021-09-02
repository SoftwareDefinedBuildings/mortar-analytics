import time
import os
import json
import warnings
import pandas as pd
from os.path import join


class SaveBacnetPoints(object):
    """
    This class creates methods to save historical data directly from
    the BACnet network.
    """

    def __init__(self, bacnet_info, bacpypesAPP, reading_rate=5*60, saving_rate=15*60, archive_length=36*3600,
                timezone=None, prj_folder="./", data_folder="DATA", data_file="data_measurements"):

        self.bacnet_info    = bacnet_info
        self.bacpypesAPP    = bacpypesAPP
        self.reading_rate   = reading_rate
        self.saving_rate    = saving_rate
        self.timezone       = timezone
        self.prj_folder     = prj_folder
        self.data_folder    = data_folder
        self.data_path      = join(prj_folder, data_folder)
        self.data_file      = join(self.data_path, f"{data_file}.json")
        self.max_records    = archive_length/float(reading_rate)
        self.last_saving    = 0

        # set timezone
        if self.timezone is None:
            warnings.warn(f"Timezone not set explicity! Check timezone variable.\
                Default to computer defaults: {time.tzname}")
        else:
            self.set_timezone(self.timezone)

        # create data directory
        self.make_folder(self.data_path)

        # initialize records with metadata
        self.init_sensor_records()


    def init_sensor_records(self):
        """
        Initialize sensor records with relevant metadata
        """

        # check if sensor records file exist
        if os.path.exists(self.data_file):
            sensor_records = self.read_json(self.data_file)
        else:

            sensor_records = {}

            for ids, row in self.bacnet_info.iterrows():
                sensor_id = int(row["bacnet_instance"])
                sensor_name = str(row["point_name"]).split("#")[-1]
                sensor_unit = str(row["unit"]).split("/")[-1]

                sensor_records[sensor_id] = {
                    "name": sensor_name,
                    "unit": sensor_unit,
                    "readings": []
                }

            # save sensor records
            self.save_json(self.data_file, sensor_records)

        # save sensor records in class
        self.sensor_records = sensor_records


    def save_json(self, path, data, mode="w"):
        """
        Save measurement data to path in json format.

        Parameters
        ----------
        path : str
            Path name to where data will be saved
        data : dict
            Dictionary object of measurement data to save. First level key is 
            the sensor id.
        mode : str, optional
            Define as 'a' to append to record keeping file or 'w' to overwrite e.g. a temporary file.
        """

        with open(path, mode=mode) as f:
            json.dump(data, f)
        f.close()


    def read_json(self, path):
        """
        Read measurement data contained in json format
        """

        with open(path, "r") as f:
            data = json.load(f)
        f.close()

        return data


    def set_timezone(self, timezone):
        """
        Set timezone on your local computer

        Set user define timezone on your local computer. This timezone will be 
        reflected in the sensor measurement recordings.

        Parameters
        ----------
        timezone : str
            The name of the timezone to set.
        """

        if hasattr(time, 'tzset'):
            os.environ['TZ'] = timezone
            time.tzset()
            print(f"Timezone was set to {time.tzname}")
        else:
            warnings.warn(f"Cannot set timezone on this machine directly! \
                Current computer timezone is {time.tzname}")


    def make_folder(self, path):
        """
        Make a new folder if it does not exist.

        Parameters
        ----------
        path : str
            Path name to new folder.
        """

        if not os.path.exists(path):
            os.makedirs(path)


    def convert_to_int(self, sensor_value):

        if isinstance(sensor_value, str):
            if sensor_value.lower() in ['active', 'on']:
                sensor_value = 1.0
            elif sensor_value.lower() in ['inactive', 'off']:
                sensor_value = 0.0

        return sensor_value


    def save_sensor_records(self):
        """
        Save latest update of sensor records
        """

        self.save_json(self.data_file, self.sensor_records)


    def get_sensor_value(self, print_vals=False):
        """
        Get current value from sensor in a BACnet network
        """
        read_attr = "presentValue"

        current_readings = []

        for ids, row in self.bacnet_info.iterrows():
            obj_type = row['bacnet_type']
            obj_inst = row['bacnet_instance']
            bacnet_addr = row['bacnet_addr']

            bacnet_args = [bacnet_addr, obj_type, obj_inst, read_attr]

            reading_time = time.time()
            sensor_value = self.bacpypesAPP.read_prop(bacnet_args)

            if print_vals:
                print(f"{reading_time}: {sensor_value}\n")

            current_readings.append((int(obj_inst), reading_time, self.convert_to_int(sensor_value)))

        return current_readings


    def archive_sensor_values(self):
        """
        Get current sensor readings and archive readings
        """

        current_readings = self.get_sensor_value()

        for sensor_id, reading_time, sensor_val in current_readings:
            # check if max records has been reached, if yes then remove oldest record
            if len(self.sensor_records[sensor_id]["readings"]) > self.max_records:
                removed_record = self.sensor_records[sensor_id]["readings"].pop(0)

            self.sensor_records[sensor_id]["readings"].append((reading_time, self.convert_to_int(sensor_val)))

        # save sensor records
        if abs(time.time() - self.last_saving) >= self.saving_rate:
            self.save_sensor_records()

            # reset last saving time
            self.last_saving = time.time()


    def records_to_df(self):
        """
        Convert json sensor records to pandas dataframe
        """
        dfs = {}

        for sensor_id in self.sensor_records.keys():
            idx = []
            val = []
            sensor_name = self.sensor_records[sensor_id]["name"]
            for reading_time, sensor_value in self.sensor_records[sensor_id]["readings"]:
                readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reading_time))
                idx.append(readable_time)
                val.append(sensor_value)

            dfs[sensor_id] = pd.Series(data=val, index=idx, name=sensor_name)

        return dfs
