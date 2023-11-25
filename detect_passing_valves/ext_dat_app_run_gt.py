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
from itertools import combinations
from copy import deepcopy

from app import _analyze_vlv, check_folder_exist, clean_final_report
from plot_data import *

# from app import _make_tdiff_vs_aflow_plot, \
#     analyze_timestamps, rename_existing, drop_unoccupied_dat, calc_long_t_diff, \
#     build_logistic_model, find_bad_vlv_operation, print_passing_mgs, _make_tdiff_vs_vlvpo_plot


def read_multi_csvs(discharge_temp_file, airflow_rate_file, vlv_pos_file, room_temp_file, dat_folder):
    """
    Read and combine multiple csv files which contain the same
    data stream types within one files e.g. all discharge air temps
    within 1 file.
    """

    discharge_temp = pd.read_csv(discharge_temp_file, index_col=0, parse_dates=True)
    vlv_pos = pd.read_csv(vlv_pos_file, index_col=0, parse_dates=True)
    airflow = pd.read_csv(airflow_rate_file, index_col=0, parse_dates=True)
    rm_temp = pd.read_csv(room_temp_file, index_col=0, parse_dates=True)

    col_labels = {
        "dnstream_ta": [vav.split(":")[0] for vav in discharge_temp.columns.values],
        "vlv_po": [vav.split(":")[0] for vav in vlv_pos.columns.values],
        "air_flow": [vav.split(":")[0] for vav in airflow.columns.values],
        "rm_temp": [vav.split(":")[0] for vav in rm_temp.columns.values],
    }

    dat_stream_map = {
        "dnstream_ta": discharge_temp,
        "vlv_po": vlv_pos,
        "air_flow": airflow,
        "rm_temp": rm_temp,
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

    # remove complete data from dictionary container
    complete_vav = []
    for vav in full_col_names_matched_streams.keys():
        if len(full_col_names_matched_streams[vav]) == len(dat_streams_list.keys()):
            complete_vav.append(full_col_names_matched_streams[vav])

    #######
    ## Combine each vavs data streams
    #######
    vavs = {}
    for i, vav in matched_streams_summary.iterrows():
        vav_name = vav.name
        vlv_dat = pd.DataFrame()
        for stream in vav.keys():
            # subset specific stream
            cur_col = vav[stream]
            if pd.isna(cur_col):
                continue
            cur_df = dat_stream_map[stream].loc[:, cur_col]
            # make sure it is numeric
            cur_df = pd.to_numeric(cur_df, errors='coerce')
            cur_df = cur_df.to_frame(name=stream)

            # add it to the container
            vlv_dat = vlv_dat.merge(cur_df, how='outer', left_index=True, right_index=True)

        # add vav data to container
        vavs[vav_name] = {
            'vlv_dat': vlv_dat,
            'row': {
                'vlv': 'vlv_' + vav_name,
                'site': 'lion',
                'equip': vav_name,
                'upstream_type': None,
            }
        }

    return vavs


def merge_down_up_stream_dat(vavs, ahu_file, matched_ahu_vav_file):

    ahu_df = pd.read_csv(ahu_file, index_col=0, parse_dates=True)
    matched_df = pd.read_csv(matched_ahu_vav_file, index_col=0, parse_dates=True)

    update_stream = 'upstream_ta'

    for i, vav in matched_df.iterrows():
        vav_name = vav.name
        upstream_ta = vav[update_stream]
        if pd.isna(upstream_ta):
            continue
        

        upstream_ta_df = ahu_df.loc[:, upstream_ta + '.SA.T']
        upstream_ta_df = pd.to_numeric(upstream_ta_df, errors='coerce')
        upstream_ta_df = upstream_ta_df.to_frame(name=update_stream)

        # get timestamp interval from existing streams
        vlv_dat = vavs[vav_name]['vlv_dat']

        vlv_dat = vlv_dat.merge(upstream_ta_df, how='outer', left_index=True, right_index=True)
        vlv_dat = vlv_dat.resample('15min').mean()

        vavs[vav_name]['vlv_dat'] = vlv_dat

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
    dat_folder = join('./', 'external_data', 'bldg_gt_pr', '20211118')
    project_folder = join('./', 'external_analysis', 'bldg_gt_pr', 'lg_4hr_shrt_1hr_test_no_aflw_req_10C_threshold_2nd_draft_revisit_hiquality_plots')

    # read files
    discharge_temp_file = join(dat_folder, 'B44-B45 Discharge Air Temp Sensor Readings - 01MAY2021 to 10NOV2021.csv')
    airflow_rate_file = join(dat_folder, 'B44-B45 Air Volume Sensor Readings - 01MAY2021 to 10NOV2021.csv')
    vlv_pos_file = join(dat_folder, 'INC5088495 VLV CMD.csv')
    room_temp_file = join(dat_folder, 'B44-B45 Room Temp Sensor Readings - 01MAY2021 to 10NOV2021.csv')

    ahu_file = join(dat_folder, 'zooms-20211111-13.10.16.csv')
    matched_ahu_vav_file = join(dat_folder, 'matched_streams_summary_with_ahu.csv')

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

    vavs_df = read_multi_csvs(discharge_temp_file, airflow_rate_file, vlv_pos_file, room_temp_file, dat_folder)
    vavs_df = merge_down_up_stream_dat(vavs_df, ahu_file, matched_ahu_vav_file)

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

        # define variables
        vlv_dat = dict(row)
        # run passing valve detection algorithm
        vlv_df, passing_type = _analyze_vlv(vlv_df, row, th_bad_vlv=TH_BAD_VLV, th_time=12, project_folder=project_folder, detection_params=detection_params)
        vavs_df[key]['vlv_dat'] = vlv_df

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

    # create timeseries plots of the data
    plot_fault_valves(post_process_vlv_dat, fault_dat_path, fig_folder_faults, time_format="Timestamp('%Y-%m-%d %H:%M:%S')")
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
    avail_cols = list(set(vav_results.columns).intersection(set(numeric_cols)))
    vav_results[avail_cols] = vav_results[avail_cols].apply(pd.to_numeric, errors='coerce')

    na_folder = vav_results['folder'].isna()
    vav_results.loc[~na_folder,'folder_short'] = vav_results.loc[~na_folder, 'folder'].apply(os.path.basename)

    # summary statistics for each site
    vav_results_grp = vav_results.groupby(['site', 'folder_short'])
    vav_results_grp['long_t'].describe()

    import pdb; pdb.set_trace()