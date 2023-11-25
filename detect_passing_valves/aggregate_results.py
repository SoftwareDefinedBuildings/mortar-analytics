from os.path import join
import os
import pandas as pd
import numpy as np

def parse_dict_list_file(line):

    dictionary = dict()
    pairs = line.strip().strip(",").strip('{}').split(', ')
    for pr in pairs:
        pair = pr.split(': ')
        dictionary[pair[0].strip('\'\'\"\"')] = pair[1].strip('\'\'\"\"')

    return dictionary

def separate_heat_transfer_col(df, col):
    heat_tfr_pwer = []
    heat_tfr_enrg = []
    for hl in df[col]:
        if pd.notnull(hl):
            hl = hl.strip('()').split(", ")
            heat_tfr_pwer.append(float(hl[0]))
            heat_tfr_enrg.append(float(hl[1]))
        else:
            heat_tfr_pwer.append(np.nan)
            heat_tfr_enrg.append(np.nan)

    return heat_tfr_pwer, heat_tfr_enrg

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
    heat_loss_pwer, heat_loss_enrg = separate_heat_transfer_col(df=all_datasets, col='heat_loss_pwr-avg_nrgy-sum')

    all_datasets['heat_loss_pwr'] = heat_loss_pwer
    all_datasets['heat_loss_enrg'] = heat_loss_enrg

    # separate intentional heat rate
    heat_intend_pwer, heat_intend_enrg = separate_heat_transfer_col(df=all_datasets, col='heat_intentional_pwr-avg_nrgy-sum')

    all_datasets['heat_intend_pwr'] = heat_intend_pwer
    all_datasets['heat_intend_enrg'] = heat_intend_enrg


    # summary statistics on long term difference for each site
    site_grp = all_datasets.groupby(['site'])
    folder_grp = all_datasets.groupby(['folder_short'])
    vav_results_grp = all_datasets.groupby(['site', 'folder_short'])
    grp_dat_stats = vav_results_grp['long_t'].describe()
    folder_grp['long_t'].describe()

    # summary statistics on long term difference for all data
    agg_dat_grp = all_datasets.groupby(['folder_short'])
    agg_dat_stats = agg_dat_grp['long_t'].describe()


    # summary statistics on heat loss due to passing valves
    no_sensor_faults = all_datasets.loc[all_datasets["folder_short"] != "sensor_fault"]
    folder_grp_nsf = no_sensor_faults.groupby(['folder_short'])
    heat_loss = folder_grp_nsf["heat_loss_enrg"].sum().sum()
    heat_intent = folder_grp_nsf["heat_intend_enrg"].sum().sum()

    heat_loss_ratio = heat_loss/heat_intent

    # on aggregate
    no_sensor_faults["long_term_fail_num_times_detected"].describe()
    no_sensor_faults["long_term_fail_avg_minutes"].describe()

    no_sensor_faults["short_term_fail_num_times_detected"].describe()
    no_sensor_faults["short_term_fail_avg_minutes"].describe()

    no_sensor_faults["heat_loss_pwr"].describe()
    no_sensor_faults["heat_loss_enrg"].sum()/1000

    # by site
    from_btuhr_to_watts = 0.293071
    site_grp_nsf = no_sensor_faults.groupby(['site'])

    site_heat_loss = site_grp_nsf["heat_loss_enrg"].sum()
    site_heat_intend = site_grp_nsf["heat_intend_enrg"].sum()

    site_heat_loss/site_heat_intend

    # by fault category
    site_grp_nsf["heat_loss_pwr"].describe()*from_btuhr_to_watts

    # all
    all_datasets.loc[all_datasets["folder_short"] == "bad_valves", "heat_loss_enrg"]
    all_datasets.loc[:,"heat_intend_enrg"]

    all_datasets["heat_loss_pwr"].describe()*from_btuhr_to_watts

    subset_lion = np.logical_and(all_datasets["site"] == "lion", all_datasets["folder_short"] == "bad_valves")
    df_bad_lion = all_datasets.loc[subset_lion]

    df_bad_lion["loss_pct"] = df_bad_lion["heat_loss_enrg"]/df_bad_lion["heat_intend_enrg"]
    df_bad_lion.sort_values("loss_pct", ascending=False)


