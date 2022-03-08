import pandas as pd
import numpy as np
import re
import os
import random

from os.path import join

from bokeh.io import show, save, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, LinearAxis, Range1d, BoxAnnotation
from bokeh.plotting import figure


def parse_list_timestamps(str_ts_list, tz='UTC'):
    """
    Convert a list of pandas timestamps represented as a string to a
    readable format i.e. pandas timestamp object
    """
    # convert string dates to readable time stamps
    fault_dates = str_ts_list.strip('][')
    fault_dates = re.split(r'(?<=\)\)), ', fault_dates)
    f_dates = [re.split(r'(?<=\)), ', ts[1:-1]) for ts in fault_dates]

    f_dts = [pd.to_datetime(ts, format="Timestamp('%Y-%m-%d %H:%M:%S%z', tz='{}')".format(tz)) for ts in f_dates]

    return f_dts


def plot_valve_data(csv_path, fault_dates=None, fig_folder='./'):
    """
    Plot valve data from mortar
    """

    # read csv
    vlv_dat = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # define plot data and parameters
    y_overlimit = 0.05
    left_y_max = np.ceil(max(vlv_dat['upstream_ta'].max(), vlv_dat['dnstream_ta'].max()))
    left_y_min = np.floor(min(vlv_dat['upstream_ta'].min(), vlv_dat['dnstream_ta'].min()))

    right_sec_y_max = np.ceil(vlv_dat['temp_diff'].max())
    right_sec_y_min = np.floor(vlv_dat['temp_diff'].min())

    sub_cols = ['upstream_ta', 'dnstream_ta', 'vlv_po', 'temp_diff']
    if 'air_flow' in vlv_dat.columns:
        right_y_max = np.ceil(vlv_dat['air_flow'].max())
        right_y_min = np.floor(vlv_dat['air_flow'].min())
        sub_cols.append('air_flow')

    src = ColumnDataSource(vlv_dat.loc[:, sub_cols])

    # make the plot
    max_date_idx = min(480, len(vlv_dat.index))
    p = figure(plot_height=300, plot_width=800, tools='xpan', toolbar_location=None,
                x_axis_type='datetime', x_axis_location='above',
                x_range=(vlv_dat.index[0], vlv_dat.index[max_date_idx]),
                y_range = Range1d(start=left_y_min*(1-y_overlimit), end=left_y_max*(1+y_overlimit)),
                background_fill_color='#ffffff'
                )
    p.yaxis.axis_label = 'Temperature [F]'

    # highlight problem areas
    if fault_dates is not None:
        fault_hilight = []
        for ts in fault_dates:
            box_ann = BoxAnnotation(left=ts[0], right=ts[1], fill_color='#db8370', fill_alpha=0.15)
            fault_hilight.append(box_ann)
            p.add_layout(box_ann)

    # line plots
    p.step('index', 'upstream_ta', source=src, color='#7093db', line_width=2)
    p.step('index', 'dnstream_ta', source=src, color='#db7093', line_width=2)

    if 'air_flow' in vlv_dat.columns:
        p.extra_y_ranges = {"vlvPos": Range1d(start=-1, end=101),
                            "airFlow": Range1d(start=right_y_min*(1-y_overlimit), end=right_y_max*(1+y_overlimit))
                            }
        p.add_layout(LinearAxis(y_range_name='airFlow', axis_label='Air flow [cfm]'), 'right')
        p.step('index', 'air_flow', source=src, color='#93db70', y_range_name='airFlow', line_width=2)
    else:
        p.extra_y_ranges = {"vlvPos": Range1d(start=-1, end=101)}

    p.add_layout(LinearAxis(y_range_name='vlvPos', axis_label='Valve position [%]'), 'left')
    p.step('index', 'vlv_po', source=src, color='#9a9a9a', line_width=0.5, line_dash='4 4', y_range_name='vlvPos')

    # range selector tool
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select = figure(title="Orange highlight area represent passing valve operation",
                    plot_height=100, plot_width=800, 
                    y_range=Range1d(start=-1, end=101),
                    x_axis_type="datetime",
                    tools="", toolbar_location=None, background_fill_color="#ffffff")
    select.yaxis.axis_label = 'Valve'

    select.step('index', 'vlv_po', source=src, color='#70dbb8')
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    select.extra_y_ranges = {"tempDiff": Range1d(start=right_sec_y_min*(1-y_overlimit), end=right_sec_y_max*(1+y_overlimit))}
    select.add_layout(LinearAxis(y_range_name='tempDiff', axis_label='TDiff'), 'right')
    select.step('index', 'temp_diff', source=src, color='#b870db', y_range_name='tempDiff')

    if fault_dates is not None:
        for box_ann in fault_hilight:
            select.add_layout(box_ann)

    # save plot
    head, tail = os.path.split(csv_path)
    plot_name = '{}-timeseries.html'.format(tail.split('.csv')[0])
    output_file(join(fig_folder, plot_name))
    save(column(p, select))


def plot_fault_valves(vlv_dat_folder, fault_dat_path, fig_folder):
    """
    Plot timeseries data for valves that were detected as passing valves
    """

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # get file paths of csvs
    all_csv_files = os.listdir(vlv_dat_folder)

    # read csv file with detected passing valves
    fault_dat = pd.read_csv(fault_dat_path, index_col=False)

    for idx, df_row in fault_dat.iterrows():
        if pd.notnull(df_row['long_term_fail_str_end_dates']):
            fault_dates = df_row['long_term_fail_str_end_dates']
            fault_dates = parse_list_timestamps(fault_dates)
            vlv_name = "{}-{}-{}".format(df_row['site'], df_row['equip'], df_row['vlv'])
            csv_names = [f for f in all_csv_files if vlv_name in f]

            for csv in csv_names:
                # plot fault data
                csv_path = join(vlv_dat_folder, csv)
                plot_valve_data(csv_path, fault_dates, fig_folder)

    print('-------Finished processing passing valve plots-----')


def plot_good_valves(vlv_dat_folder, sample_size, fig_folder):
    """
    Plot timeseries data for valves that have normal operation
    """
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # get file paths of csvs
    all_csv_files = os.listdir(vlv_dat_folder)
    good_valve_files = os.listdir(join(vlv_dat_folder, '../', 'good_valves'))
    good_valve_files = [tail.split('.png')[0] for tail in good_valve_files]

    # sample define number of files
    sample_files = random.sample(good_valve_files, sample_size)

    for sf in sample_files:
        # plot fault data
        csv_path = join(vlv_dat_folder,'{}_dat.csv'.format(sf))
        if os.path.exists(csv_path):
            plot_valve_data(csv_path, fig_folder=fig_folder)
        else:
            print('{} valve csv file not found'.format(sf))

    print('-------Finished processing good valve plots-----')


if __name__ == '__main__':

    # define data sources
    project_folder = join("./", "with_airflow_checks_year_end")
    vlv_dat_folder = join(project_folder, "csv_data")

    # fault data plots
    fig_folder_faults = join(project_folder, 'timeseries_valve_faults')
    fault_dat_path = join(project_folder, "passing_valve_results.csv")

    plot_fault_valves(vlv_dat_folder, fault_dat_path, fig_folder_faults)

    # good data plots
    fig_folder_good = join(project_folder, 'timeseries_valve_good')
    plot_good_valves(vlv_dat_folder, sample_size=20, fig_folder=fig_folder_good)


