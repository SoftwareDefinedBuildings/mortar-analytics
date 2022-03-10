from os.path import join
import os
from unicodedata import numeric
import pandas as pd
import numpy as np

def parse_dict_list_file(line):

    dictionary = dict()
    pairs = line.strip().strip(",").strip('{}').split(', ')
    for pr in pairs:
        pair = pr.split(': ')
        dictionary[pair[0].strip('\'\'\"\"')] = pair[1].strip('\'\'\"\"')

    return dictionary

if __name__ == '__main__':
    dat_folder = join('with_airflow_checks_year_start', 'csv_data')
    project_folder = join('./', 'external_analysis')

    mortar_dat_file = join(project_folder, 'MORTAR', 'lg_4hr_shrt_1hr_test_no_aflw_req_10C_threshold')
    bear_dat_file = join(project_folder, 'bldg_trc_rs', 'lg_4hr_shrt_1hr_test_no_aflw_req_10C_threshold')
    lion_dat_file = join(project_folder, 'bldg_gt_pr', 'lg_4hr_shrt_1hr_test_no_aflw_req_10C_threshold')

    all_datasets = []
    for dataset in [mortar_dat_file, bear_dat_file, lion_dat_file]:
        # Perform additional analysis
        f = open(join(dataset, 'minimum_airflow_values.txt'), 'r')
        lines = f.readlines()
        f.close()

        vav_results = []
        for line in lines:
            vav_results.append(parse_dict_list_file(line))

        vav_results = pd.DataFrame.from_records(vav_results)
        all_datasets.append(vav_results)

    all_datasets = pd.concat(all_datasets).reset_index(drop=True)

    # clean dataframe
    numeric_cols = ['minimum_air_flow_cutoff', 'long_t', 'long_tbad', 'bad_ratio', 'long_to']
    avail_cols = list(set(all_datasets.columns).intersection(set(numeric_cols)))
    all_datasets[avail_cols] = all_datasets[avail_cols].apply(pd.to_numeric, errors='coerce')

    all_datasets = all_datasets.dropna(subset=['folder'])

    all_datasets['folder_short'] = all_datasets['folder'].apply(os.path.basename)


    # add rest of info including heat loss due to passing valves
    all_passing_valve_results = []
    for dataset in [mortar_dat_file, bear_dat_file, lion_dat_file]:
        final_df = pd.read_csv(join(dataset, 'passing_valve_results.csv'), index_col=0)
        all_passing_valve_results.append(final_df)

    all_passing_valve_results = pd.concat(all_passing_valve_results).reset_index(drop=True)

    all_datasets = pd.merge(all_datasets, all_passing_valve_results, how='left', on=['vlv', 'site', 'equip'])

    # separate heat rate loss from energy loss
    heat_loss_pwer = []
    heat_loss_enrg = []
    for hl in all_datasets['heat_loss_pwr-avg_nrgy-sum']:
        if pd.notnull(hl):
            hl = hl.strip('()').split(", ")
            heat_loss_pwer.append(float(hl[0]))
            heat_loss_enrg.append(float(hl[1]))
        else:
            heat_loss_pwer.append(np.nan)
            heat_loss_enrg.append(np.nan)
    
    all_datasets['heat_loss_pwr'] = heat_loss_pwer
    all_datasets['heat_loss_enrg'] = heat_loss_enrg


    # summary statistics on long term difference for each site
    site_grp = all_datasets.groupby(['site'])
    vav_results_grp = all_datasets.groupby(['site', 'folder_short'])
    grp_dat_stats = vav_results_grp['long_t'].describe()

    # summary statistics on long term difference for all data
    agg_dat_grp = all_datasets.groupby(['folder_short'])
    agg_dat_stats = agg_dat_grp['long_t'].describe()


    # summary statistics on heat loss due to passing valves
    
    # on aggregate
    agg_dat_grp["long_term_fail_num_times_detected"].describe()
    agg_dat_grp["long_term_fail_avg_minutes"].describe()

    agg_dat_grp["short_term_fail_num_times_detected"].describe()
    agg_dat_grp["short_term_fail_avg_minutes"].describe()

    # by site
    from_btuhr_to_watts = 0.293071
    site_grp["heat_loss_pwr"].describe()*from_btuhr_to_watts
    all_datasets["heat_loss_pwr"].describe()*from_btuhr_to_watts



