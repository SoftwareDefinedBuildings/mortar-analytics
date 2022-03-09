"""
Run the detect passing valve algorithm on external building data from Mortar.
Data was downloaded before if was down.
The data needed to run application are the following:
    timestamp
    upstream vav air temperature
    downstream vav air temperature
    vav valve position
    vav airflow rate
"""

import pandas as pd
import pickle
import numpy as np
import os
import glob
from os.path import join

from app import _analyze_vlv, check_folder_exist, clean_final_report
from plot_data import *

def read_files(folder, sample_num=None):
    """
    Read individual files that contain all datastream from one VAV
    """
    analysis_cols = ['upstream_ta', 'dnstream_ta', 'vlv_po', 'air_flow']

    vav_files = glob.glob(join(folder, "*.csv"))

    if sample_num is not None:
        from random import sample
        vav_files = sample(vav_files, sample_num)

    vavs = {}
    for vfile in vav_files:
        cur_vav_name = os.path.basename(vfile).split(".csv")[0].replace('_dat', '')

        try:
            site, vav, vlv = cur_vav_name.split("-", 2)
        except ValueError:
            import pdb; pdb.set_trace()

        vlv_dat = pd.read_csv(vfile, index_col=0, parse_dates=True)

        # check that the columns are available
        df_cols = vlv_dat.columns
        avail_cols = [ac for ac in analysis_cols if ac in df_cols]

        vlv_dat = vlv_dat.loc[:, avail_cols]

        # add vav data to container
        vavs[cur_vav_name] = {
            'vlv_dat': vlv_dat,
            'row': {
                'vlv': vlv,
                'site': site,
                'equip': vav,
                'upstream_type': None,
            }
        }
    return vavs


def parse_dict_list_file(line):

    dictionary = dict()
    pairs = line.strip().strip(",").strip('{}').split(', ')
    for pr in pairs:
        pair = pr.split(': ')
        dictionary[pair[0].strip('\'\'\"\"')] = pair[1].strip('\'\'\"\"')

    return dictionary

if __name__ == '__main__':
    dat_folder = join('with_airflow_checks_year_start', 'csv_data')
    project_folder = join('./', 'external_analysis', 'MORTAR', 'lg_4hr_shrt_1hr_test_no_aflw_req_10C_threshold')

    # define container folders
    good_folder = 'good_valves'         # name of path to the folder to save the plots of the correct operating valves
    bad_folder = 'bad_valves'           # name of path to the folder to save the plots of the malfunction valves
    sensor_fault_folder = 'sensor_fault'# name of path to the folder to save plots of equipment with sensor faults
    air_flow_folder = 'air_flow_plots'  # name of path to the folder to save plots of the air flow values
    csv_folder = 'csv_data'             # name of path to the folder to save detailed valve data

    # check if holding folders exist
    check_folder_exist(join(project_folder, bad_folder))
    check_folder_exist(join(project_folder, good_folder))
    check_folder_exist(join(project_folder, sensor_fault_folder))
    check_folder_exist(join(project_folder, air_flow_folder))
    check_folder_exist(join(project_folder, csv_folder))

    # define user parameters
    TH_BAD_VLV = 10
    detection_params = {
        "th_bad_vlv": TH_BAD_VLV,           # temperature difference from long term temperature difference to consider an operating point as malfunctioning
        "th_time": 12,             # length of time, in minutes, after the valve is closed to determine if valve operating point is malfunctioning
        "long_term_fail": 4*60,    # number of minutes to trigger an long-term passing valve failure
        "shrt_term_fail": 60,      # number of minutes to trigger an intermitten passing valve failure
        "th_vlv_fail": 20,         # equivalent percentage of valve open for determining failure.
        "air_flow_required": False, # boolean indicated is air flow rate data should strictly be used.
        "af_accu_factor": 0.60,
        "good_folder": good_folder,
        "bad_folder": bad_folder,
        "sensor_fault_folder": sensor_fault_folder,
        "air_flow_folder": air_flow_folder,
        "csv_folder": csv_folder,
    }

    vavs_df = read_files(dat_folder)

    results = []
    vav_count_summary = []
    n_skipped = 0
    for key in vavs_df.keys():
        cur_vlv_df = vavs_df[key]['vlv_dat']
        required_streams = [stream in cur_vlv_df.columns for stream in ['dnstream_ta', 'upstream_ta', 'vlv_po']]
        if not all(required_streams):
            n_skipped += 1
            print("Skipping VAV = {} because all required streams are not available".format(key))
            continue

        vlv_df = vavs_df[key]['vlv_dat']
        row = vavs_df[key]['row']
        vav_count_summary.append({'site': row['site'], 'equip': row['equip']})

        # define variables
        vlv_dat = dict(row)

        # run passing valve detection algorithm
        passing_type = _analyze_vlv(vlv_df, row, th_bad_vlv=TH_BAD_VLV, th_time=12, project_folder=project_folder, detection_params=detection_params)

        # save results
        vlv_dat.update(passing_type)
        results.append(vlv_dat)


    # report and plot
    # define fault folders
    print("Skipped a total of {} terminal units".format(n_skipped))
    fault_dat_path = join(project_folder, "passing_valve_results.csv")
    fig_folder_faults = join(project_folder, "ts_valve_faults")
    fig_folder_good = join(project_folder, "ts_valve_good")
    post_process_vlv_dat = join(project_folder, "csv_data")
    vav_count_file = join(project_folder, 'vav_count_summary.csv')
    raw_analyzed_data = join(project_folder, 'raw_analyzed_data.pkl')
    raw_analyzed_results = join(project_folder, 'raw_analyzed_results.pkl')

    final_df = pd.DataFrame.from_records(results)
    final_df = clean_final_report(final_df, drop_null=False)
    final_df.to_csv(fault_dat_path)

    vav_count_summary = pd.DataFrame.from_records(vav_count_summary)
    vav_count_summary.to_csv(vav_count_file)

    raw_df = open(raw_analyzed_data, "wb")
    pickle.dump(vavs_df, raw_df)
    raw_df.close()

    raw_result = open(raw_analyzed_results, "wb")
    pickle.dump(results, raw_result)
    raw_result.close()

    # raw_df = open(raw_analyzed_data, "rb")
    # object_file = pickle.load(raw_df)
    # raw_df.close()

    # create timeseries plots of the data
    plot_fault_valves(post_process_vlv_dat, fault_dat_path, fig_folder_faults, time_format="Timestamp('%Y-%m-%d %H:%M:%S%z', tz='UTC')")
    plot_valve_ts_streams(post_process_vlv_dat, join(project_folder, sensor_fault_folder), sample_size='all', fig_folder=fig_folder_faults)

    # plot good vav operation timeseries
    plot_valve_ts_streams(post_process_vlv_dat, join(project_folder, good_folder), sample_size='all', fig_folder=fig_folder_good)

    # Perform additional analysis
    f = open(join(project_folder, 'minimum_airflow_values.txt'), 'r')
    lines = f.readlines()
    f.close()

    vav_results = []
    for line in lines:
        vav_results.append(parse_dict_list_file(line))

    vav_results = pd.DataFrame.from_records(vav_results)

    numeric_cols = ['minimum_air_flow_cutoff', 'long_t', 'long_tbad', 'bad_ratio', 'long_to']
    vav_results[numeric_cols] = vav_results[numeric_cols].apply(pd.to_numeric, errors='coerce')
    vav_results['folder_short'] = vav_results['folder'].apply(os.path.basename)

    # summary statistics for each site
    vav_results_grp = vav_results.groupby(['site', 'folder_short'])
    vav_results_grp['long_t'].describe()


    import pdb; pdb.set_trace()