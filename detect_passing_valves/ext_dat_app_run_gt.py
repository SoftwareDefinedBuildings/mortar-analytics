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

from os.path import join
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from itertools import combinations
from copy import deepcopy

from app import _analyze_vlv, check_folder_exist, clean_final_report

# from app import _make_tdiff_vs_aflow_plot, \
#     analyze_timestamps, rename_existing, drop_unoccupied_dat, calc_long_t_diff, \
#     build_logistic_model, find_bad_vlv_operation, print_passing_mgs, _make_tdiff_vs_vlvpo_plot


def read_multi_csvs(discharge_temp_file, airflow_rate_file, vlv_pos_file):
    """
    Read and combine multiple csv files which contain the same
    data stream types within one files e.g. all discharge air temps
    within 1 file.
    """

    discharge_temp = pd.read_csv(discharge_temp_file, index_col=0, parse_dates=True)
    vlv_pos = pd.read_csv(vlv_pos_file, index_col=0, parse_dates=True)
    airflow = pd.read_csv(airflow_rate_file, index_col=0, parse_dates=True)

    col_labels = {
        "dnstream_ta": [vav.split(":")[0] for vav in discharge_temp.columns.values],
        "vlv_po": [vav.split(":")[0] for vav in vlv_pos.columns.values],
        "air_flow": [vav.split(":")[0] for vav in airflow.columns.values],
    }

    dat_stream_map = {
        "dnstream_ta": discharge_temp,
        "vlv_po": vlv_pos,
        "air_flow": airflow,
    }


    # start to match columns
    dat_streams_list = col_labels
    dat_stream_comb = list(combinations(dat_streams_list, 2))
    matched_streams = {}

    for i, dat_comb in enumerate(dat_stream_comb):
        for j, col in enumerate(col_labels[dat_comb[0]]):
            if col in matched_streams.keys():
                matched_streams[col].update({dat_comb[0]: j})
            else:
                matched_streams[col] = {dat_comb[0]: j}
            key = dat_comb[1]
            if col in col_labels[key]:
                match_stream = [(vav, k) for k, vav in enumerate(col_labels[key]) if col == vav]
                if len(match_stream) == 1:
                    matched_streams[match_stream[0][0]].update({key: match_stream[0][1]})
                else:
                    print("More than two streams matched")
                    import pdb; pdb.set_trace()
        
        # remove complete data from dictionary container
        complete_vav = []
        for vav in matched_streams.keys():
            if len(matched_streams[vav]) == 3:
                complete_vav.append(matched_streams[vav])

    # retreive full column names based on index of column
    full_col_names_matched_streams = deepcopy(matched_streams)
    for vav in matched_streams:
        for stream in matched_streams[vav].keys():
            col_idx = matched_streams[vav][stream]
            full_col_name = dat_stream_map[stream].columns[col_idx]
            full_col_names_matched_streams[vav][stream] = full_col_name

    matched_streams_summary = pd.DataFrame.from_records(full_col_names_matched_streams).transpose()
    matched_streams_summary = matched_streams_summary.sort_values(by=list(col_labels.keys()))
    matched_streams_summary.to_csv(join(dat_folder, 'matched_streams_summary.csv'), index_label='vav_label')

    tags = ['Equip Fail', 'Scan Off']

    ## Alternative way to match columns

    # # figure out which file has the most columns
    # max_label = (0, '')
    # for key in col_labels:
    #     col_len = len(col_labels[key])
    #     if col_len > max_label[0]:
    #         max_label = (col_len, key)

    # matched_streams = []
    # other_streams = [key for key in col_labels.keys() if key not in max_label[1]]
    # for i, col in enumerate(col_labels[max_label[1]]):
    #     cur_vav_unit = [(max_label[1], col, i)]
    #     for key in other_streams:
    #         if col in col_labels[key]:
    #             match_stream = [(vav, j) for j, vav in enumerate(col_labels[key]) if col == vav]
    #             if len(match_stream) == 1:
    #                 cur_vav_unit.append((key, match_stream[0][0], match_stream[0][1]))
    #             else:
    #                 print("More than two streams matched")
    #                 import pdb; pdb.set_trace()
    #     matched_streams.append(cur_vav_unit)

    # # determine which streams have completed data
    # complete_vav = []
    # for vav in matched_streams:
    #     if len(vav) == 3:
    #         complete_vav.append(vav)
    #         for stream in vav:
    #             col_labels[stream[0]].remove(stream[1])






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
                'site': 'bldg_trc_rs',
                'equip': vav,
                'upstream_type': None,
            }
        }

    return vavs


def calc_add_features(vav_df, drop_na=False):
    """
    Calculate additional features needed for application
    """
    # identify when valve is open
    vav_df['vlv_open'] = vav_df['vlv_po'] > 0

    # calculate temperature difference between downstream and upstream air
    vav_df['temp_diff'] = vav_df['dnstream_ta'] - vav_df['upstream_ta']

    # drop na
    if drop_na:
        vav_df = vav_df.dropna()

    # drop values where vav supply air is less than ahu supply air
    vav_df = vav_df[vav_df['temp_diff'] >= 0]

    return vav_df


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


if __name__ == '__main__':
    dat_folder = join('./', 'external_data', 'bldg_gt_pr')
    project_folder = join('./', 'external_analysis', 'bldg_gt_pr', 'lg_4hr_shrt_1hr_test_no_off_period')

    # read files
    discharge_temp_file = join(dat_folder, 'B44-B45 Discharge Air Temp Sensor Readings - 01MAY2021 to 04NOV2021.csv')
    airflow_rate_file = join(dat_folder, 'INC5088495 AIR VOL.csv')
    vlv_pos_file = join(dat_folder, 'INC5088495 VLV CMD.csv')

    # define container folders
    good_folder = 'good_valves'         # name of path to the folder to save the plots of the correct operating valves
    bad_folder = 'bad_valves'           # name of path to the folder to save the plots of the malfunction valves
    air_flow_folder = 'air_flow_plots'  # name of path to the folder to save plots of the air flow values
    csv_folder = 'csv_data'             # name of path to the folder to save detailed valve data

    # check if holding folders exist
    check_folder_exist(join(project_folder, bad_folder))
    check_folder_exist(join(project_folder, good_folder))
    check_folder_exist(join(project_folder, air_flow_folder))
    check_folder_exist(join(project_folder, csv_folder))

    # define user parameters
    detection_params = {
        "th_bad_vlv": 5,           # temperature difference from long term temperature difference to consider an operating point as malfunctioning
        "th_time": 12,             # length of time, in minutes, after the valve is closed to determine if valve operating point is malfunctioning
        "window": 15,              # aggregation window, in minutes, to average the raw measurement data
        "long_term_fail": 4*60,    # number of minutes to trigger an long-term passing valve failure
        "shrt_term_fail": 60,      # number of minutes to trigger an intermitten passing valve failure
        "th_vlv_fail": 20,         # equivalent percentage of valve open for determining failure.
        "good_folder": good_folder,
        "bad_folder": bad_folder,
        "air_flow_folder": air_flow_folder,
        "csv_folder": csv_folder,
    }

    df = read_multi_csvs(discharge_temp_file, airflow_rate_file, vlv_pos_file)

    vavs_df = clean_df(df)

    results = []
    for key in vavs_df.keys():
        vavs_df[key]['vlv_dat'] = calc_add_features(vavs_df[key]['vlv_dat'])
        vlv_df = vavs_df[key]['vlv_dat']
        row = vavs_df[key]['row']

        # remove data when heat system was off
        off_str = '08/31/2021'
        off_end = '10/07/2021'

        vlv_df = exclude_time_interval(vlv_df, off_str, off_end)

        # define variables
        vlv_dat = dict(row)
        # run passing valve detection algorithm
        passing_type = _analyze_vlv(vlv_df, row, th_bad_vlv=5, th_time=12, window=5, project_folder=project_folder, detection_params=detection_params)

        # save results
        vlv_dat.update(passing_type)
        results.append(vlv_dat)

    final_df = pd.DataFrame.from_records(results)
    final_df = clean_final_report(final_df, drop_null=False)
    final_df.to_csv(join(project_folder, "passing_valve_results.csv"))

    import pdb; pdb.set_trace()