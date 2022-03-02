"""
Run the detect passing valve algorithm on external building data.
The data needed to run application are the following:
    timestamp
    upstream vav air temperature
    downstream vav air temperature
    vav valve position
    vav airflow rate
"""

import pandas as pd
import numpy as np
import os
import time
import pickle

from os.path import join

from app import _analyze_vlv, check_folder_exist, clean_final_report
from plot_data import *

# from app import _make_tdiff_vs_aflow_plot, \
#     analyze_timestamps, rename_existing, drop_unoccupied_dat, calc_long_t_diff, \
#     build_logistic_model, find_bad_vlv_operation, print_passing_mgs, _make_tdiff_vs_vlvpo_plot


def read_multi_csvs(csv_list):
    """
    Read and combine multiple csv file containing the same
    data streams e.g. 1 file per month, year, or other time interval
    """

    dfs = []
    for csv_file in csv_list:
        csv = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        # verify that columns are numeric
        for col in csv.columns:
            csv.loc[:, col] = pd.to_numeric(csv.loc[:, col], errors="coerce")

        dfs.append(csv)

    df_merge = pd.concat(dfs)

    # aggregate any repeated timestamps
    df_merge = df_merge.sort_index()
    df_merge = df_merge.groupby(level=0).mean()

    return df_merge


def clean_df(df):
    """
    prepare data for the algorithm
    """
    vav_datastream_labels  = df.columns[4:]

    vav_equip = [lab.split("/")[0] for lab in vav_datastream_labels.values]
    stream_type = [lab.split("/")[1] for lab in vav_datastream_labels.values]

    # map to standard names in the app
    com_stream_map = {
        'upstream_ta': 'ahu-3/sa_temp',
        'oat': 'OA Temp'
    }

    ind_stream_map = {
        'dnstream_ta': 'da_temp',
        'vlv_po': 'hw_valve',
        'air_flow': 'flow_tn',
        'zone_temp': 'zone_temp'
    }

    vavs = {}
    for vav in np.unique(vav_equip):
        common_keys = list(com_stream_map.keys())
        vlv_dat = df.loc[:, com_stream_map[common_keys[0]]]
        vlv_dat.name = common_keys[0]

        if len(common_keys) > 1:
            for com_pt in common_keys[1:]:
                vlv_dat = pd.concat([vlv_dat, df.loc[:, com_stream_map[com_pt]]], axis=1)
                vlv_dat = vlv_dat.rename(columns={com_stream_map[com_pt]: com_pt})

        for stream in ind_stream_map.keys():
            vav_stream = '/'.join([vav, ind_stream_map[stream]])
            if vav_stream in vav_datastream_labels.values:
                new_stream = df.loc[:, vav_stream]

                vlv_dat = pd.concat([vlv_dat, new_stream], axis=1)
                vlv_dat = vlv_dat.rename(columns={vav_stream: stream})

        # verify that all necessary data points are available
        stream_avail = [col in vlv_dat.columns for col in ['upstream_ta', 'dnstream_ta', 'vlv_po']]

        if not all(stream_avail):
            print(f"VAV={vav} does not have all required data streams\n")
            print(f"Missing {3 - np.count_nonzero(stream_avail)} streams, please check.\n")
            continue

        # save in a dictionary
        vavs[vav] = {
            'vlv_dat': vlv_dat,
            'row': {
                'vlv': 'vlv_' + vav,
                'site': 'bear',
                'equip': vav,
                'upstream_type': None,
            }
        }

    return vavs


def CountFrequency(my_list):
     
    # Creating an empty dictionary
    freq = {}
    for items in np.unique(my_list):
        freq[items] = my_list.count(items)

    return freq


def exclude_time_interval(df, int_str, int_end):
    """
    Exclude time interval where data is not representative
    """

    time_interval = pd.to_datetime([int_str, int_end])
    within_interval = np.logical_and(df.index > time_interval[0], df.index < time_interval[1])

    return df.loc[~within_interval, :]


def parse_dict_list_file(line):

    dictionary = dict()
    pairs = line.strip().strip(",").strip('{}').split(', ')
    for pr in pairs:
        pair = pr.split(': ')
        dictionary[pair[0].strip('\'\'\"\"')] = pair[1].strip('\'\'\"\"')

    return dictionary

if __name__ == '__main__':
    # NOTE: HW plant off from August 31 to October 7 2021
    dat_folder = join('./', 'external_data', 'bldg_trc_rs')
    project_folder = join('./', 'external_analysis', 'bldg_trc_rs', 'lg_4hr_shrt_1hr_test_no_aflw_req')

    csv_list = [
        join(dat_folder, 'zone trends, September 2021.csv'),
        join(dat_folder, 'zone trends, October 2021.csv'),
        join(dat_folder, 'Schoellkopf zone trends 20211103_A.csv'),
        join(dat_folder, 'Schoellkopf zone trends 20211103_B.csv'),
        join(dat_folder, 'Schoellkopf zone trends 20211109.csv'),
        join(dat_folder, 'Schoellkopf zone trends Nov2021 downloaded 20211122.csv')
    ]


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
    detection_params = {
        "th_bad_vlv": 10,           # temperature difference from long term temperature difference to consider an operating point as malfunctioning
        "th_time": 12,             # length of time, in minutes, after the valve is closed to determine if valve operating point is malfunctioning
        "long_term_fail": 4*60,    # number of minutes to trigger an long-term passing valve failure
        "shrt_term_fail": 60,      # number of minutes to trigger an intermitten passing valve failure
        "th_vlv_fail": 20,         # equivalent percentage of valve open for determining failure.
        "air_flow_required": False, # boolean indicated is air flow rate data should strictly be used.
        "good_folder": good_folder,
        "bad_folder": bad_folder,
        "sensor_fault_folder": sensor_fault_folder,
        "air_flow_folder": air_flow_folder,
        "csv_folder": csv_folder,
    }

    df = read_multi_csvs(csv_list)

    vavs_df = clean_df(df)

    results = []
    vav_count_summary = []
    for key in vavs_df.keys():
        vlv_df = vavs_df[key]['vlv_dat']
        row = vavs_df[key]['row']
        vav_count_summary.append({'site': row['site'], 'equip': row['equip']})

        # remove data when heat system was off
        off_str = '08/31/2021'
        off_end = '10/07/2021'

        vlv_df = exclude_time_interval(vlv_df, off_str, off_end)

        # define variables
        vlv_dat = dict(row)
        # run passing valve detection algorithm
        passing_type = _analyze_vlv(vlv_df, row, th_bad_vlv=10, th_time=12, project_folder=project_folder, detection_params=detection_params)

        # save results
        vlv_dat.update(passing_type)
        results.append(vlv_dat)

    # report and plot
    # define fault folders
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

    # create timeseries plots of the data
    plot_fault_valves(post_process_vlv_dat, fault_dat_path, fig_folder_faults, time_format="Timestamp('%Y-%m-%d %H:%M:%S')")
    plot_valve_ts_streams(post_process_vlv_dat, join(project_folder, sensor_fault_folder), sample_size='all', fig_folder=fig_folder_faults)

    # plot good vav operation timeseries
    plot_valve_ts_streams(post_process_vlv_dat, join(project_folder, good_folder), sample_size='all', fig_folder=fig_folder_good)

    import pdb; pdb.set_trace()
    # Perform additional analysis
    f = open(join(project_folder, 'minimum_airflow_values.txt'), 'r')
    lines = f.readlines()
    f.close()

    vav_results = []
    for line in lines:
        vav_results.append(parse_dict_list_file(line))

    vav_results = pd.DataFrame.from_records(vav_results)

    numeric_cols = ['minimum_air_flow_cutoff', 'long_t', 'long_tbad', 'bad_ratio', 'long_to']
    avail_cols = list(set(vav_results.columns).intersection(set(numeric_cols)))
    vav_results[avail_cols] = vav_results[avail_cols].apply(pd.to_numeric, errors='coerce')

    vav_results['folder_short'] = vav_results['folder'].apply(os.path.basename)

    # summary statistics for each site
    vav_results_grp = vav_results.groupby(['site', 'folder_short'])
    vav_results_grp['long_t'].describe()

    import pdb; pdb.set_trace()