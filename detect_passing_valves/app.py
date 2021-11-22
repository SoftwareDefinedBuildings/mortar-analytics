import pymortar
import sys
import pandas as pd
import numpy as np
import os
import time

from os.path import join
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

def _query_and_qualify():
    """
    Build query to return control valves, up- and down- stream air temperatures relative to the
    valve, and other related data. Then qualify which sites can run this app.

    Parameters
    ----------
    None

    Returns
    -------
    query: dictionary containing query, sites, and qualify response

    """

    # connect to client
    client = pymortar.Client()

    # initialize container for query information
    query = dict()

    # define query to analyze VAV valves
    vav_query = """SELECT *
    WHERE {
        ?equip        rdf:type/rdfs:subClassOf?   brick:VAV .
        ?equip        bf:isFedBy+                 ?ahu .
        ?vlv          rdf:type                    ?vlv_type .
        ?ahu          bf:hasPoint                 ?upstream_ta .
        ?equip        bf:hasPoint                 ?dnstream_ta .
        ?upstream_ta  rdf:type/rdfs:subClassOf*   brick:Supply_Air_Temperature_Sensor .
        ?dnstream_ta  rdf:type/rdfs:subClassOf*   brick:Supply_Air_Temperature_Sensor .
        ?equip        bf:hasPoint                 ?vlv .
        ?vlv          rdf:type/rdfs:subClassOf*   brick:Valve_Command .
    };"""

    fan_query = """SELECT *
    WHERE {
        ?equip        rdf:type/rdfs:subClassOf?   brick:VAV .
        ?equip        bf:hasPoint                 ?air_flow .
        ?air_flow     rdf:type/rdfs:subClassOf*   brick:Supply_Air_Flow_Sensor .
    };"""

    # define queries to analyze AHU valves
    ahu_sa_query = """SELECT *
    WHERE {
        ?vlv        rdf:type/rdfs:subClassOf*   brick:Valve_Command .
        ?vlv        rdf:type                    ?vlv_type .
        ?equip      bf:hasPoint                 ?vlv .
        ?equip      rdf:type/rdfs:subClassOf*   brick:Air_Handling_Unit .
        ?air_temps  rdf:type/rdfs:subClassOf*   brick:Supply_Air_Temperature_Sensor .
        ?equip      bf:hasPoint                 ?air_temps .
        ?air_temps  rdf:type                    ?temp_type .
    };"""

    ahu_ra_query = """SELECT *
    WHERE {
        ?vlv        rdf:type/rdfs:subClassOf*   brick:Valve_Command .
        ?vlv        rdf:type                    ?vlv_type .
        ?equip      bf:hasPoint                 ?vlv .
        ?equip      rdf:type/rdfs:subClassOf*   brick:Air_Handling_Unit .
        ?air_temps  rdf:type/rdfs:subClassOf*   brick:Return_Air_Temperature_Sensor .
        ?equip      bf:hasPoint                 ?air_temps .
        ?air_temps  rdf:type                    ?temp_type .
    };"""

    # find sites with these sensors and setpoints
    qualify_vav_resp = client.qualify([vav_query])
    qualify_sa_resp = client.qualify([ahu_sa_query])
    qualify_ra_resp = client.qualify([ahu_ra_query])
    qualify_fan_resp = client.qualify([fan_query])

    if qualify_vav_resp.error != "":
        print("ERROR: ", qualify_vav_resp.error)
        sys.exit(1)
    elif len(qualify_vav_resp.sites) == 0:
        print("NO SITES RETURNED")
        sys.exit(0)

    vav_sites = qualify_vav_resp.sites
    ahu_sites = np.intersect1d(qualify_sa_resp.sites, qualify_ra_resp.sites)
    tlt_sites = np.union1d(vav_sites, ahu_sites)
    print("running on {0} sites".format(len(tlt_sites)))

    # save queries
    query['query'] = dict()
    query['query']['vav'] = vav_query
    query['query']['ahu_sa'] = ahu_sa_query
    query['query']['ahu_ra'] = ahu_ra_query
    query['query']['air_flow'] = fan_query

    # save qualify responses
    query['qualify'] = dict()
    query['qualify']['vav'] = qualify_vav_resp
    query['qualify']['ahu_sa'] = qualify_sa_resp
    query['qualify']['ahu_ra'] = qualify_ra_resp
    query['qualify']['air_flow'] = qualify_fan_resp

    # save sites
    query['sites'] = dict()
    query['sites']['vav'] = vav_sites
    query['sites']['ahu'] = ahu_sites
    query['sites']['tlt'] = tlt_sites

    return query


def _fetch(query, eval_start_time, eval_end_time, window=15):
    """
    Build the fetch query and define the time interval for analysis

    Parameters
    ----------
    query: dictionary containing query, sites, and qualify response

    eval_start_time : start date and time in format (yyyy-mm-ddTHH:MM:SSZ) for the thermal
                      comfort evaluation period

    eval_end_time : end date and time in format (yyyy-mm-ddTHH:MM:SSZ) for the thermal
                    comfort evaluation period

    window : aggregation window, in minutes, to average the raw measurement data


    Returns
    -------
    fetch_resp : Mortar FetchResponse object

    """

    # connect to client
    client = pymortar.Client()

    # build the fetch request for the vav valves
    vav_request = pymortar.FetchRequest(
        sites=query['sites']['vav'],
        views=[
            pymortar.View(
                name="dnstream_ta",
                definition=query['query']['vav'],
            ),
            pymortar.View(
                name="air_flow",
                definition=query['query']['air_flow']
            ),
        ],
        dataFrames=[
            pymortar.DataFrame(
                name="vlv",
                aggregation=pymortar.MEAN,
                window="15m",
                timeseries=[
                    pymortar.Timeseries(
                        view="dnstream_ta",
                        dataVars=["?vlv"],
                    )
                ]
            ),
            pymortar.DataFrame(
                name="air_flow",
                aggregation=pymortar.MEAN,
                window="15m",
                timeseries=[
                    pymortar.Timeseries(
                        view="air_flow",
                        dataVars=["?air_flow"],
                    )
                ]
            ),
            pymortar.DataFrame(
                name="dnstream_ta",
                aggregation=pymortar.MEAN,
                window="15m",
                timeseries=[
                    pymortar.Timeseries(
                        view="dnstream_ta",
                        dataVars=["?dnstream_ta"],
                    )
                ]
            ),
            pymortar.DataFrame(
                name="upstream_ta",
                aggregation=pymortar.MEAN,
                window="15m",
                timeseries=[
                    pymortar.Timeseries(
                        view="dnstream_ta",
                        dataVars=["?upstream_ta"],
                    )
                ]
            ),
        ],
        time=pymortar.TimeParams(
            start=eval_start_time,
            end=eval_end_time,
        )
    )


    # build the fetch request for the ahu valves
    ahu_request = pymortar.FetchRequest(
        sites=query['sites']['ahu'],
        views=[
            pymortar.View(
                name="dnstream_ta",
                definition=query['query']['ahu_sa'],
            ),
            pymortar.View(
                name="upstream_ta",
                definition=query['query']['ahu_ra'],
            ),
        ],
        dataFrames=[
            pymortar.DataFrame(
                name="ahu_valve",
                aggregation=pymortar.MEAN,
                window="15m",
                timeseries=[
                    pymortar.Timeseries(
                        view="dnstream_ta",
                        dataVars=["?vlv"],
                    )
                ]
            ),
            pymortar.DataFrame(
                name="dnstream_ta",
                aggregation=pymortar.MEAN,
                window="15m",
                timeseries=[
                    pymortar.Timeseries(
                        view="dnstream_ta",
                        dataVars=["?air_temps"],
                    )
                ]
            ),
            pymortar.DataFrame(
                name="upstream_ta",
                aggregation=pymortar.MEAN,
                window="15m",
                timeseries=[
                    pymortar.Timeseries(
                        view="upstream_ta",
                        dataVars=["?air_temps"],
                    )
                ]
            ),
        ],
        time=pymortar.TimeParams(
            start=eval_start_time,
            end=eval_end_time,
        )
    )

    # call the fetch api for VAV data
    fetch_resp_vav = client.fetch(vav_request)

    print("-----Dataframe for VAV valves-----")
    print(fetch_resp_vav)
    print(fetch_resp_vav.view('dnstream_ta'))

    # call the fetch api for AHU data
    fetch_resp_ahu = client.fetch(ahu_request)
    ahu_metadata = reformat_ahu_view(fetch_resp_ahu)

    print("-----Dataframe for AHU valves-----")
    print(fetch_resp_ahu)
    print(ahu_metadata)

    # save fetch responses
    fetch_resp = dict()
    fetch_resp['vav'] = fetch_resp_vav
    fetch_resp['ahu'] = fetch_resp_ahu

    return fetch_resp


def reformat_ahu_view(fetch_resp_ahu):
    """
    Rename, reformat, and delete cooling valves from ahu metadata

    Parameters
    ----------
    fetch_resp_ahu : Mortar FetchResponse object for AHU data

    Returns
    -------
    ahu_metadata: Pandas object with AHU metadata and no valves used for cooling

    """
    # supply air temp metadata
    ahu_sa = fetch_resp_ahu.view('dnstream_ta')
    ahu_sa = ahu_sa.rename(columns={'air_temps': 'dnstream_ta', 'temp_type': 'dnstream_ta', 'air_temps_uuid': 'dnstream_ta uuid'})

    # return air temp metadata
    ahu_ra = fetch_resp_ahu.view('upstream_ta')
    ahu_ra = ahu_ra.rename(columns={'air_temps': 'upstream_ta', 'temp_type': 'upstream_type', 'air_temps_uuid': 'upstream_ta uuid'})

    # join supply and return air temperature data into on dataset
    ahu_metadata = ahu_sa.merge(ahu_ra, on=['vlv', 'equip', 'vlv_type', 'site'], how='inner')

    # delete cooling valve commands
    heat_vlv = [x not in ['Cooling_Valve_Command'] for x in ahu_metadata['vlv_type']]

    return ahu_metadata[heat_vlv]


def _clean_vav(fetch_resp_vav, row):
    """
    Make a pandas dataframe with relavent vav data for the specific valve 
    and clean from NA values. Calculate temperature difference between
    downstream and upstream air temperatures.

    Parameters
    ----------
    fetch_resp_vav : Mortar FetchResponse object for the vav data

    row: Pandas series object with metadata for the specific vav valve

    Returns
    -------
    vav_df: Pandas dataframe with valve timeseries data

    """

    df_flow = get_vav_flow(fetch_resp_vav, row, fillna=None)

    # combine data points in one dataframe
    vav_sa = fetch_resp_vav['dnstream_ta'][row['dnstream_ta_uuid']]
    ahu_sa = fetch_resp_vav['upstream_ta'][row['upstream_ta_uuid']]
    vlv_po = fetch_resp_vav['vlv'][row['vlv_uuid']]

    if df_flow is not None:
        vav_df = pd.concat([ahu_sa, vav_sa, vlv_po, df_flow], axis=1)
        vav_df.columns = ['upstream_ta', 'dnstream_ta', 'vlv_po', 'air_flow']

    else:
        vav_df = pd.concat([ahu_sa, vav_sa, vlv_po], axis=1)
        vav_df.columns = ['upstream_ta', 'dnstream_ta', 'vlv_po']

    # identify when valve is open
    vav_df['vlv_open'] = vav_df['vlv_po'] > 0

    # calculate temperature difference between downstream and upstream air
    vav_df['temp_diff'] = vav_df['dnstream_ta'] - vav_df['upstream_ta']

    # drop na
    # vav_df = vav_df.dropna()

    # drop values where vav supply air is less than ahu supply air
    vav_df = vav_df[vav_df['temp_diff'] >= 0]

    return vav_df


def drop_unoccupied_dat(df, occ_str=6, occ_end=18, wkend_str=5, air_flow_required=False):
    """
    Drop data rows from dataframe for timeseries that are during unoccupied hours. Uses airflow
    data if available else it uses building occupancy hours.

    Parameters
    ----------
    df: Pandas dataframe object with timeseries data

    occ_str: float number indicating start of building occupancy

    occ_end: float number indicating end of building occupancy

    wkend_str: int number indicating start of weekend. 5 indicates Saturday and 6 indicates Sunday

    Returns
    -------
    df: Pandas dataframe with data values during building occupancy hours
    """

    if 'air_flow' in df.columns:
        # drop values where there is no air flow
        xs, ys = density_data(df['air_flow'], rescale_dat=df['temp_diff'])
        min_idx = return_extreme_points(ys, type_of_extreme='min', sort=False)

        if min_idx is not None:
            min_air_flow = xs[min_idx[0]]
        else:
            min_air_flow = np.percentile(xs, 5)

        df = df.loc[df['air_flow'] > min_air_flow]
    elif 'air_flow' not in df.columns and air_flow_required:
        df = pd.DataFrame()
    else:
        # drop values outside occupancy hours
        print("No airflow data, using explicit occupancy hours to do analysis.")
        df = occupied_hours_subset(df, occ_str, occ_end, wkend_str)

    return df


def get_vav_flow(fetch_resp_vav, row, fillna=None):
    """
    Return VAV supply air flow

    Parameters
    ----------
    fetch_resp_vav : Mortar FetchResponse object for the vav data

    row: Pandas series object with metadata for the specific vav valve

    fillna: Method to use for filling na values in dataframe. Options: backfill, bfill, pad, ffill, None

    Returns
    -------
    df_flow: Pandas dataframe with vav supply air flow timeseries data
    """

    # fine corresponding air flow sensor for vav
    flow_view = fetch_resp_vav.view('air_flow')
    flow_meta = flow_view.loc[np.logical_and(flow_view['equip'] == row['equip'], flow_view['site'] == row['site'])]

    fidx = 0
    if flow_meta.shape[0] > 1:
        print("Multiple airflow sensors found for VAV {} in site {}! \
               Please check. Will continue using the first sensor.".format(row['equip'], row['site']))
        print(flow_meta)
        time.sleep(3)
    if flow_meta.shape[0] == 0:
        return None

    # return air flow timeseries data
    flow_id = flow_meta.loc[flow_meta.index[fidx], 'air_flow_uuid']
    df_flow = fetch_resp_vav['air_flow'].loc[:, flow_id]

    if fillna is not None:
        # fill na values
        df_flow = df_flow.fillna(method=fillna)

    return df_flow


def occupied_hours_subset(df, occ_str, occ_end, wkend_str=5, timestamp_col=None):
    """
    Returns data containing values during building occupancy of Pandas DataFrame

    Parameters
    ----------
    df: Pandas dataframe object with timeseries data

    occ_str: float number indicating start of building occupancy

    occ_end: float number indicating end of building occupancy

    wkend_str: int number indicating start of weekend. 5 indicates Saturday and 6 indicates Sunday

    timestamp_col: If timeseries object is not defined in the index of the pandas dataframe then 
            input the column name containing the timeseries

    Returns
    -------
    df_is_occupied: Pandas dataframe with data values during building occupancy hours
    """
    # define the timeseries data
    if timestamp_col is None:
        df_ts = df.index
    else:
        df_ts = df[timestamp_col]

    bool_str_hr = (df_ts.hour + df_ts.minute/60.0) >= occ_str
    bool_end_hr = (df_ts.hour + df_ts.minute/60.0) <= occ_end
    bool_is_weekday = df_ts.weekday < 5 # 5 and 6 are Sat and Sun, respectively

    is_occupied = np.logical_and(bool_str_hr, bool_end_hr, bool_is_weekday)

    return df[is_occupied]


def _clean_ahu(fetch_resp_ahu, row):
    """
    Make a pandas dataframe with relavent ahu data for the specific valve 
    and clean from NA values.

    Parameters
    ----------
    fetch_resp_ahu : Mortar FetchResponse object for the AHU data

    row: Pandas series object with metadata for the specific vav valve

    Returns
    -------
    ahu_df: Pandas dataframe with valve timeseries data

    """
    dnstream = fetch_resp_ahu['dnstream_ta'][row['dnstream_ta uuid']]
    upstream = fetch_resp_ahu['upstream_ta'][row['upstream_ta uuid']]

    vlv_po = fetch_resp_ahu['ahu_valve'][row['vlv_uuid']]

    ahu_df = pd.concat([upstream, dnstream, vlv_po], axis=1)
    ahu_df.columns = ['upstream_ta', 'dnstream_ta', 'vlv_po']

    # identify when valve is open
    ahu_df['vlv_open'] = ahu_df['vlv_po'] > 0

    # calculate temperature difference between downstream and upstream air
    ahu_df['temp_diff'] = ahu_df['dnstream_ta'] - ahu_df['upstream_ta']

    # drop na
    # ahu_df = ahu_df.dropna()

    # drop values where vav supply air is less than ahu supply air
    #ahu_df = ahu_df[ahu_df['temp_diff'] >= 0]

    return ahu_df


######
# define tools
# TODO: Separate the tools into a new python file
######

def scale_0to1(vals):
    """
    Scale pandas series object data from 0 to 1

    Parameters
    ----------
    vals: Pandas series object or Pandas dataframe colum to scale from 0 to 1.

    Returns
    -------
    scaled_vals: Pandas series object with values scaled from 0 to 1
    """

    max_val = vals.max()
    min_val = vals.min()

    scaled_vals = (vals - min_val) / (max_val - min_val)

    return scaled_vals


def rescale_fit(scaled_vals, vals=None, max_val=None, min_val=None):
    """
    Rescale values of pandas series that are 0 to 1 to match the interval
    of another pandas series object values

    Parameters
    ----------
    scaled_vals: Pandas series object with values scaled from 0 to 1 and needs to be unscaled

    vals: Pandas series object or Pandas dataframe colum with unnormalized values.
        This is used to extract max and min to unscaled the scaled_vals.

    max_val: a float number indicating the maximum value to rescale vector. Must
        also define min_val.

    min_val: a float number indicating the minimum value to rescale vector. Must
        also define max_val.

    Returns
    -------
    unscaled_vals: Pandas series object of unscaled values
    """

    if vals is not None:
        max_val = vals.max()
        min_val = vals.min()
    elif (max_val is None) or (min_val is None):
        raise ValueError('Need to define vals dataframe or both maximum and minimum values for rescale!')

    unscaled_vals = min_val + scaled_vals*(max_val - min_val)

    return unscaled_vals


def sigmoid(x, k, x0):
    """
    Sigmoid function curve to do a logistic model

    Parameters
    ----------
    x: independent variable
    k: slope of the sigmoid function
    x0: midpoint/inflection point of the sigmoid function

    Returns
    -------
    y: value of the function at point x
    """
    return 1.0 / (1 + np.exp(-k * (x - x0)))

def build_logistic_model(df, x_col='vlv_po', y_col='temp_diff'):
    """
    Build a logistic model with data provided

    Parameters
    ----------
    df: Pandas dataframe object with x and y variables to make model

    x_col: column name that contains x, independent, variable

    y_col: column name that contains y, dependent, variable

    Returns
    -------
    df_fit: Pandas dataframe object with y_fitted values to a logistic model

    popt: an array of the optimized parameters, slope and inflection point of the sigmoid function
    """

    try:
        # fit the curve
        scaled_pos = scale_0to1(df[x_col])
        scaled_t = scale_0to1(df[y_col])
        popt, pcov = curve_fit(sigmoid, scaled_pos, scaled_t)

        # calculate fitted temp difference values
        est_k, est_x0 = popt
        popt[1] = rescale_fit(popt[1], df[x_col])
        y_fitted = rescale_fit(sigmoid(scaled_pos, est_k, est_x0), df[y_col])
        y_fitted.name = 'y_fitted'

        # sort values
        df_fit = pd.concat([df[x_col], y_fitted], axis=1)
        df_fit = df_fit.sort_values(by=x_col)
    except RuntimeError:
        print("Model unabled to be developed\n")
        return None, None

    return df_fit, popt

def try_limit_dat_fit_model(vlv_df, df_fraction):
    # calculate fit model
    nrows, ncols = vlv_df.shape
    some_pts = np.random.choice(nrows, int(nrows*df_fraction))
    try:
        df_fit, popt = build_logistic_model(vlv_df.iloc[some_pts])
    except RuntimeError:
        try:
            df_fit, popt = build_logistic_model(vlv_df)
        except RuntimeError:
            print("No regression found")
            df_fit = None
    return df_fit


def check_folder_exist(folder):
    """
    Check the existance of the defined folder. If it does
    not exist, then create folder.

    Parameters
    ----------
    folder: name of path to check its existance

    Returns
    -------
    None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def calc_long_t_diff(vlv_df):
    """
    Calculate statistics on difference between down- and up-
    stream temperatures to determine the long term temperature difference
    when valve is closed.

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    Returns
    -------
    long_t: dictionary object with statisitics of temperature difference
    """

    if vlv_df is None:
        return None

    df_vlv_close = vlv_df.loc[np.logical_and(vlv_df['cons_ts_vlv_c'], vlv_df['steady'])]
    if df_vlv_close is None:
        return None

    long_t = df_vlv_close['temp_diff'].describe()

    return long_t

def analyze_timestamps(vlv_df, th_time, window, row=None, project_folder='./'):
    """
    Analyze timestamps and valve operation in a pandas dataframe to determine which row values 
    are th_time minutes after a changed state e.g. determine which data corresponds
    to steady-state and transient values.

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    th_time: length of time, in minutes, after the valve is closed to determine if
        valve operating point is malfunctioning e.g. allow enough time for residue heat to
        dissipate from the coil. Recommended time for reheat coils > 12 minutes.

    window : aggregation window, in minutes, to average the raw measurement data

    row: Pandas series object with metadata for the specific vav valve

    project_folder: name of path for the project and used to save the plot.

    Returns
    -------
    vav_df: same input pandas dataframe but with added columns indicating:
        cons_ts: boolean indicating consecutive timestamps
        cons_ts_vlv_c: boolean indicating consecutive timestamps when valve is commanded closed
        same: boolean indicating group number of cons_ts_vlv_c
        steady: boolean indicating if the timestamp is in steady state condition
    """

    min_ts = int(th_time/window) + (th_time % window > 0)
    min_tst = pd.Timedelta(th_time, unit='min')

    # only get consecutive timestamps datapoints
    ts = pd.Series(vlv_df.index)
    ts_int = pd.Timedelta(window, unit='min')
    cons_ts = ((ts - ts.shift(-1)).abs() <= ts_int) | (ts.diff() <= ts_int)

    if (len(cons_ts) < min_ts) | ~(np.any(cons_ts)):
        return None

    vlv_df.loc[:, 'cons_ts'] = np.array(cons_ts)
    vlv_df.loc[:, 'cons_ts_vlv_c'] = np.logical_and(~vlv_df['vlv_open'], vlv_df['cons_ts'])
    vlv_df.loc[:, 'same'] = vlv_df['cons_ts_vlv_c'].astype(int).diff().ne(0).cumsum()

    # subset by consecutive times that exceed th_time
    lal = vlv_df.groupby('same')

    steady = []
    for grp in lal.groups.keys():
        for ts in lal.groups[grp]:
            init_ts = lal.groups[grp][0]
            steady.append(init_ts+min_tst < ts)

    vlv_df.loc[:, 'steady'] = np.array(steady)

    # save csv data if row is defined
    if row is not None:
        _name = "{}-{}-{}_dat".format(row['site'], row['equip'], row['vlv'])

        full_path = rename_existing(join(project_folder, csv_folder, _name + '.csv'), idx=0, row=row)
        vlv_df.to_csv(full_path)

    # drop rows of data where valve position is unknown
    vlv_df = vlv_df.dropna(subset=['vlv_po'])

    return vlv_df


def return_extreme_points(dat, type_of_extreme=None, n_modes=None, sort=True):
    """
    Return the peak and troughs of a multimodal distribution of a vector.

    Parameters
    ----------
    dat: vector of data points to develop a distribution

    type_of_extreme: type of extremes to return. If None it will return minimum
        and maximum points.

    n_modes: number of distribution peaks or troughs to return. If greater than 1
        it will return the largest or smallest.

    sort: sort the peak/trough values from smallest to largest.

    Returns
    -------
    idx: indeces of the peaks or troughs of the multimodal distribution.
    """

    a = np.diff(dat)
    asign = np.sign(a)

    signchg = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchg == 1)[0]

    # delete extreme points
    if 0 in idx:
        idx = np.delete(idx, 0)

    if (len(dat) - 1) in idx:
        idx = np.delete(idx, len(dat)-1)

    idx_num = len(idx)
    if idx_num < 2:
        return None
    else:
        type_of = []
        if dat[idx[0]] > dat[idx[1]]:
            # if true then starting inflection point is a maximum
            type_of = np.array(['max']*idx_num)
            type_of[1:][::2] = 'min'
        elif dat[idx[0]] < dat[idx[1]]:
            # if true then starting inflection point is a minimum
            type_of = np.array(['min']*idx_num)
            type_of[1:][::2] = 'max'

        # return requested inflection points
        if type_of_extreme == 'max':
            idx = idx[type_of == 'max']
        elif type_of_extreme == 'min':
            idx = idx[type_of == 'min']
        else:
            print('Returning all inflection points')

        if sort or n_modes is not None:
            idx = idx[np.argsort(dat[idx])]

        if n_modes is not None:
            if type_of_extreme == 'max':
                idx = idx[(-1*n_modes):]
            elif type_of_extreme == 'min':
                idx = idx[:n_modes]

        return idx


def _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=None, long_tbad=None, df_fit=None, bad_ratio=None, folder='./'):
    """
    Make plot showing the correct and bad operating points of the valve control along with helper annotations 
    e.g. long term average for correct and malfunction operating points when valve is commanded off, model fit, and
    bad to good operating points.

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    row: Pandas series object with metadata for the specific valve

    long_t: long-term temperature difference between down and up air streams when valve is 
            commanded close for correct operation

    long_tbad: long-term temperature difference between down and up air streams when valve is 
            commanded close for malfunction operation

    df_fit: Pandas dataframe object with y_fitted values to a logistic model

    bad_ratio: ratio showing the mulfunction operation points to good operation points

    folder: name of path to save the plot image

    Returns
    -------
    None
    """

    # plot parametes
    y_max = vlv_df['temp_diff'].max()

    good_oper_color = '#5ab300'
    bad_oper_color = '#b3005a'

    # plot temperature difference vs valve position
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.set_ylabel('Temperature difference [°F]')
    ax.set_xlabel('Valve opened [%]')
    ax.set_title("Valve = {}\nEquip. = {}".format(row['vlv'], row['equip']), loc='left')
    ax.set_ylim((0, np.ceil(y_max*1.05)))


    if any(~vlv_df['good_oper_cat']):
        ax.scatter(x=vlv_df.loc[~vlv_df['good_oper_cat'], 'vlv_po'], y=vlv_df.loc[~vlv_df['good_oper_cat'], 'temp_diff'], color = bad_oper_color, alpha=1/3, s=10, label='Pred. bad operation')

    if any(vlv_df['good_oper_cat']):
        ax.scatter(x=vlv_df.loc[vlv_df['good_oper_cat'], 'vlv_po'], y=vlv_df.loc[vlv_df['good_oper_cat'], 'temp_diff'], color = good_oper_color, alpha=1/3, s=10, label='Pred. good operation')

    # if 'color' in vlv_df.columns:
    #     ax.scatter(x=vlv_df['vlv_po'], y=vlv_df['temp_diff'], color = vlv_df['color'], alpha=1/3, s=10)
    # else:
    #     ax.scatter(x=vlv_df['vlv_po'], y=vlv_df['temp_diff'], color = '#005ab3', alpha=1/3, s=10)

    if df_fit is not None:
        # add fit line
        ax.plot(df_fit['vlv_po'], df_fit['y_fitted'], '--', label='Fitted valve model', color='#5900b3')

    if long_t is not None:
        # add long-term temperature diff
        ax.axhline(y=long_t, color='#00b3b3', label='Est. Td (closed valve-good)')

    if long_tbad is not None:
        ax.axhline(y=long_tbad, color='#ff8cc6', label='Est. Td (closed valve-bad)')

    if bad_ratio is not None:
        # add ratio where presumably passing valve
        ax.text(.2, 0.95*y_max, "Bad operation ratio={:.1f}%".format(bad_ratio))

    # legend
    # ax.legend(fontsize=8, markerscale=1, borderaxespad=0., ncol=2, loc='upper right', bbox_to_anchor=(0.15, 1.05, 1., .102))
    ax.legend(fontsize=6, markerscale=1, borderaxespad=0., ncol=2, bbox_to_anchor=(.55, 1.02), loc='lower left')

    plt_name = "{}-{}-{}".format(row['site'], row['equip'], row['vlv'])
    full_path = rename_existing(join(folder, plt_name + '.png'), idx=0, row=row)
    plt.savefig(full_path)
    plt.close()


def rename_existing(path, idx, row):
    """
    Check if the file path exists, if it does, then rename.

    Parameters
    ----------
    path: name of path to check

    idx: index of duplicate file

    row: Pandas series object with metadata for the specific valve
    """
    if os.path.exists(path):
        print('REPEATED EQUIP for {}-{}-{}'.format(row['site'], row['equip'], row['vlv']))
        idx+=1
        head, tail = os.path.split(path)
        tail = "R" + str(idx) + "-" + tail
        path = rename_existing(join(head, tail), idx, row)

    return path

def _make_tdiff_vs_aflow_plot(vlv_df, row, folder):
    """
    Create temperature difference versus air flow plots

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    row: Pandas series object with metadata for the specific valve

    folder: name of path to save the plot image

    Returns
    -------
    None
    """

    # plot parametes
    closed_vlv_color = '#640064'
    open_vlv_color = '#006400'

    vlv_df.loc[:, 'color_open'] = closed_vlv_color
    vlv_df.loc[vlv_df['vlv_open'], 'color_open'] = open_vlv_color

    # plot temperature difference vs valve position
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.set_ylabel('Temperature difference [°F]')
    ax.set_xlabel('Air flow [cfm]')
    ax.set_title("Valve = {}\nEquip. = {}".format(row['vlv'], row['equip']), loc='left')

    if any(~vlv_df['vlv_open']):
        ax.scatter(x=vlv_df.loc[~vlv_df['vlv_open'], 'air_flow'], y=vlv_df.loc[~vlv_df['vlv_open'], 'temp_diff'], color = closed_vlv_color, alpha=1/3, s=10, label='Closed valve')

    if any(vlv_df['vlv_open']):
        ax.scatter(x=vlv_df.loc[vlv_df['vlv_open'], 'air_flow'], y=vlv_df.loc[vlv_df['vlv_open'], 'temp_diff'], color = open_vlv_color, alpha=1/3, s=10, label='Open valve')
    # ax.scatter(x=vlv_df['air_flow'], y=vlv_df['temp_diff'], color = vlv_df['color_open'], alpha=1/3, s=10)

    # create density plot for air flow
    xs, ys = density_data(vlv_df['air_flow'], rescale_dat=vlv_df['temp_diff'])
    ax.plot(xs, ys)

    # find modes of the distribution and the trough before/after the modes
    max_idx = return_extreme_points(ys, type_of_extreme='max', n_modes=2)
    min_idx = return_extreme_points(ys, type_of_extreme='min', sort=False)

    if max_idx is not None:
        ax.scatter(x=xs[max_idx], y=ys[max_idx], color = '#ff0000', alpha=1, s=35)
    if max_idx is not None:
        ax.scatter(x=xs[min_idx], y=ys[min_idx], color = '#ff8000', alpha=1, s=35)

    # Legend
    ax.legend(markerscale=2)

    plt_name = "{}-{}-{}".format(row['site'], row['equip'], row['vlv'])
    full_path = rename_existing(join(folder, plt_name + '.png'), idx=0, row=row)
    plt.savefig(full_path)
    plt.close()


def density_data(dat, rescale_dat=None):
    """
    Create a kernel-density estimate using Gaussian kernels and rescale to
    match the specific valve data.

    Parameters
    ----------
    dat: vector of data points to develop a distribution

    rescale_dat: If not None, the data vector is used to determine the peak for rescaling
        the y values of the density function.
        If rescale_dat is define as 'norm', ys will be normalized from 0 to 1.

    Returns
    -------
    xs: x values of the density function

    ys: y values of the density function
    """
    #create data for density plot
    try:
        density = gaussian_kde(dat)
    except:
        print("NAs exist is airflow data. Will delete them.")
        density = gaussian_kde(dat[~dat.isna()])
    xs = np.linspace(0, max(dat), 200)

    density.covariance_factor = lambda : 0.25
    density._compute_covariance()

    # unscaled y values of density
    us_ys = density(xs)

    # rescale if rescale_dat is not None
    if isinstance(rescale_dat, (pd.Series, np.ndarray)):
        ys = rescale_fit(scale_0to1(us_ys), max_val=np.percentile(rescale_dat, 95), min_val=0)
    elif isinstance(rescale_dat, str) and 'norm' in rescale_dat:
        ys = scale_0to1(us_ys)
    else:
        ys = us_ys

    return xs, ys

def find_bad_vlv_operation(vlv_df, model, window):
    """
    Determine which timeseries values are data from probable passing valves and return 
    a pandas dataframe of only 'bad' values.

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    long_t: long-term temperature difference between down and up air streams when valve is 
            commanded close for correct operation

    window : aggregation window, in minutes, to average the raw measurement data

    Returns
    -------
    df_bad: pandas dataframe object with the time intervals that the valve is malfunctioning

    pass_type: dictionary with failure modes listed, if any. Possible failure modes are:
        long_term_fail: long term valve failure if valve seems to be passing for more than number X 
            minutes defined in global parameter 'long_term_fail'. Default 300 minutes (5 hours).
        short_term_fail: intermittent valve failure if valve seems to be passing for short periods 
            (X minutes defined in global parameter 'shrt_term_fail') due to control errors, 
            mechanical/electrical problems, or other. Default 60 minutes (1 hour).
    """

    pass_type = dict()
    hi_diff = np.percentile(vlv_df.loc[vlv_df['vlv_open'], 'temp_diff'], 95)

    if model is not None:
        vlv_po_hi_diff = model[model['y_fitted'] <= hi_diff]['vlv_po'].max()

        # define temperature difference and valve position failure thresholds
        vlv_po_th = vlv_po_hi_diff/2.0
        diff_vlv_po_th = max(model[model['vlv_po'] <= vlv_po_th]['y_fitted'].max(), 5)
    else:
        diff_vlv_po_th = max(hi_diff/4.0, 5)

    # find datapoints that exceed long-term temperature difference
    exceed_long_t = vlv_df['temp_diff'] >= diff_vlv_po_th

    # subset data by consecutive steady state values when valve is commanded closed and 
    # exceeds long-term temperature difference
    th_exceed = np.logical_and(np.logical_and(vlv_df['cons_ts_vlv_c'], vlv_df['steady']), exceed_long_t)
    df_bad = vlv_df[th_exceed]

    if df_bad.empty:
        return None, dict()

    # analyze 'bad' dataframe for possible passing valve
    bad_grp = df_bad.groupby('same')
    bad_grp_count = bad_grp['same'].count()

    # max_idx = np.argmax(bad_grp_count)
    # max_grp = bad_grp.groups[bad_grp_count.index[max_idx]]

    # if len(max_grp) > 1:
    #     max_passing_time = max_grp[-1] - max_grp[0]
    # else:
    #     max_passing_time = pd.Timedelta(0, unit='min')

    # # detect long term failures
    # if max_passing_time > pd.Timedelta(long_term_fail, unit='min'):
    #     ts_seconds = max_passing_time.seconds
    #     ts_days    = max_passing_time.days * 3600 * 24
    #     pass_type['long_term_fail'] = (ts_days+ts_seconds)/60.0

    # detect long term failures
    long_term_fail_bool = (bad_grp_count*window) > long_term_fail
    if any(long_term_fail_bool):
        long_term_fail_times = bad_grp_count[long_term_fail_bool]*window
        if long_term_fail_times.count() >= 1 or long_term_fail_times.index[-1] == vlv_df['same'].max():
            dates = [(bad_grp.groups[ky][0], bad_grp.groups[ky][-1]) for ky in long_term_fail_times.index]
            pass_type['long_term_fail'] = (long_term_fail_times.mean(), long_term_fail_times.count(), dates)

    # detect short term failures
    bad_grp_left_over = bad_grp_count[~long_term_fail_bool]
    short_term_fail_bool = (bad_grp_left_over*window) > shrt_term_fail
    if any(short_term_fail_bool):
        shrt_term_fail_times = bad_grp_left_over[short_term_fail_bool]*window
        if shrt_term_fail_times.count() >= 2 or (shrt_term_fail_times.count() >= 1 and any(long_term_fail_bool)):
            dates = [(bad_grp.groups[ky][0], bad_grp.groups[ky][-1]) for ky in shrt_term_fail_times.index]
            pass_type['short_term_fail'] = (shrt_term_fail_times.mean(), shrt_term_fail_times.count(), dates)

    return df_bad, pass_type


def print_passing_mgs(row):
    """
    Print message to user when passing valve is probable

    Parameters
    ----------
    row: Pandas series object with metadata for the specific valve

    Returns
    -------
    None
    """
    print("Probable passing valve '{}' in site {}\n".format(row['vlv'], row['site']))


def clean_final_report(final_df, drop_null=True):
    """
    Clean final report and sort by greatest number of minutes that fault was detected

    Parameters
    ----------
    final_df: pandas dataframe with valve metadata along with failure types detected

    drop_null: boolean to drop rows where no short or long term faults exist for valves.

    Returns
    -------
    final_df: cleaned and sorted report
    """
    if 'long_term_fail' in final_df.columns:
        if drop_null:
            final_df = final_df.loc[np.logical_or(~final_df['long_term_fail'].isnull(), ~final_df['short_term_fail'].isnull())]

        if 'long_term_fail' in final_df.columns:
            # separate data into multiple columns
            final_df['long_term_fail_avg_minutes'] = final_df.long_term_fail.str[0]
            final_df['long_term_fail_num_times_detected'] = final_df.long_term_fail.str[1]
            final_df['long_term_fail_str_end_dates'] = final_df.long_term_fail.str[2]

            # drop redundant columns
            final_df = final_df.drop(columns=['long_term_fail'])

        if 'short_term_fail' in final_df.columns:
            final_df['short_term_fail_avg_minutes'] = final_df.short_term_fail.str[0]
            final_df['short_term_fail_num_times_detected'] = final_df.short_term_fail.str[1]
            final_df['short_term_fail_str_end_dates'] = final_df.short_term_fail.str[2]

            # drop redundant columns
            final_df = final_df.drop(columns=['short_term_fail'])

        # sort by highest value faults
        final_df = final_df.sort_values(by=['long_term_fail_avg_minutes', 'short_term_fail_avg_minutes'], ascending=False)

    return final_df


def analyze_only_open(vlv_df, row, th_bad_vlv, project_folder):
    """
    Analyze valve data when there is only open valve data.

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    row: Pandas series object with metadata for the specific valve

    th_bad_vlv: temperature difference from long term temperature difference to consider an operating point as malfunctioning

    project_folder: name of path for the project and used to save the plot.

    Returns
    -------
    pass_type: dictionary with failure modes listed, if any. Possible failure modes are:
        non_responsive_fail: failure when the valve is open but the median temperature
            difference when the valve is command open but never goes above the above 
            the th_bad_vlv threshold.
    """
    pass_type = dict()

    long_to = vlv_df[vlv_df['vlv_open']]['temp_diff'].describe()
    if long_to['50%'] < th_bad_vlv:
        print("'{}' in site {} is open but seems to not cause an increase in air temperature\n".format(row['vlv'], row['site']))
        pass_type['non_responsive_fail'] = round(long_to['50%'] - th_bad_vlv, 2)
        folder = join(project_folder, bad_folder)
        vlv_df.loc[:, 'good_oper_cat'] = False
    else:
        vlv_df.loc[:, 'good_oper_cat'] = True
        folder = join(project_folder, good_folder)

    _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=long_to['50%'], folder=folder)

    return pass_type

def analyze_only_close(vlv_df, row, th_bad_vlv, project_folder):
    """
    Analyze valve data when there is only closed valve data.

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    row: Pandas series object with metadata for the specific valve

    th_bad_vlv: temperature difference from long term temperature difference to consider an operating point as malfunctioning

    project_folder: name of path for the project and used to save the plot.

    Returns
    -------
    pass_type: dictionary with failure modes listed, if any. Possible failure modes are:
        simple_fail: failure when the median temperature difference when the valve is commanded
            closed if above the th_bad_vlv threshold.
    """
    pass_type = dict()
    long_tc = calc_long_t_diff(vlv_df)
    if long_tc['50%'] > th_bad_vlv:
        print_passing_mgs(row)
        pass_type['simple_fail'] = round(long_tc['50%'] - th_bad_vlv, 2)
        folder = join(project_folder, bad_folder)
        import pdb; pdb.set_trace()
    else:
        vlv_df.loc[:, 'good_oper_cat'] = True
        folder = join(project_folder, good_folder)

    _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=long_tc['50%'], folder=folder)

    return pass_type

def _analyze_vlv(vlv_df, row, th_bad_vlv=5, th_time=45, window=15, project_folder='./', detection_params=None):
    """
    Analyze each valve and detect for passing valves

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    row: Pandas series object with metadata for the specific valve

    th_bad_vlv: temperature difference from long term temperature difference to consider an operating point as malfunctioning

    th_time: length of time, in minutes, after the valve is closed to determine if 
        valve operating point is malfunctioning e.g. allow enough time for residue heat to
        dissipate from the coil.

    window: aggregation window, in minutes, to average the raw measurement data

    project_folder: name of path for the project and used to save the plots and csv data.

    detection_params: dictionary of parameters that control the behavior of the application

    Returns
    -------
    None
    """

    # update variables
    if detection_params is not None:
        globals().update(detection_params)

    # container for holding types of faults
    passing_type = dict()

    # files = os.listdir(join("with_airflow_checks", bad_folder))
    # vav_oi = [entry.split("-")[1] for entry in files]
    # if row['equip'] in ['VAVRM2323']:
    #     import pdb; pdb.set_trace()

    # check for empty dataframe
    if vlv_df.empty:
        print("'{}' in site {} has no data! Skipping...".format(row['vlv'], row['site']))
        return passing_type

    if 'air_flow' in vlv_df.columns:
        # plot temp diff vs air flow
        _make_tdiff_vs_aflow_plot(vlv_df, row, folder=join(project_folder, 'air_flow_plots'))

    # Analyze timestamps and valve operation changes
    vlv_df = analyze_timestamps(vlv_df, th_time, window, row=row, project_folder=project_folder)

    if vlv_df is None:
        print("'{}' in site {} has no data after analyzing \
            consecutive timestamps! Skipping...".format(row['vlv'], row['site']))
        return passing_type

    # drop data that occurs during unoccupied hours
    vlv_df = drop_unoccupied_dat(vlv_df, occ_str=6, occ_end=18, wkend_str=5, air_flow_required=air_flow_required)

    if vlv_df.empty:
        print("'{}' in site {} has no data after hours of \
            occupancy check! Skipping...".format(row['vlv'], row['site']))
        return passing_type

    # determine if valve datastream has open and closed data
    bool_type = vlv_df['vlv_open'].value_counts().index

    if len(bool_type) < 2:
        if bool_type[0]:
            # only open valve data
            passing_type = analyze_only_open(vlv_df, row, th_bad_vlv, project_folder)
        else:
            # only closed valve data
            passing_type = analyze_only_close(vlv_df, row, th_bad_vlv, project_folder)

        return passing_type

    # TODO: Figure out what to do if long_tc is None!
    # calculate long-term temp diff when valve is closed
    long_tc = calc_long_t_diff(vlv_df)
    long_to = vlv_df[vlv_df['vlv_open']]['temp_diff'].describe()

    if long_tc is None and long_to is not None:
        pass_type = analyze_only_open(vlv_df, row, th_bad_vlv, project_folder)
        passing_type.update(pass_type)
        return passing_type

    # make simple comparison of long-term closed temp difference and user define threshold
    if long_tc['50%'] > th_bad_vlv:
        print_passing_mgs(row)
        passing_type['simple_fail'] = round(long_tc['50%'] - th_bad_vlv, 2)

    # make comparison between long-term open and long-term closed temp difference
    long_tc_to_diff = (long_tc['mean'] + long_tc['std']) - (long_to['75%'])
    if long_tc_to_diff > 0:
        print_passing_mgs(row)
        passing_type['tc_to_close_fail'] = round(long_tc_to_diff, 2)

    # assume a 0 deg difference at 0% open valve
    no_zeros_po = vlv_df.copy()
    no_zeros_po.loc[~no_zeros_po['vlv_open'], 'temp_diff'] = 0

    # make a logit regression model assuming that closed valves make a zero temp difference
    df_fit_nz, popt = build_logistic_model(no_zeros_po)

    # calculate bad valve instances vs overall dataframe
    bad_vlv, pass_type = find_bad_vlv_operation(vlv_df, df_fit_nz, window)
    passing_type.update(pass_type)

    if bad_vlv is None:
        bad_ratio = 0
        long_tbad = long_tc['mean']
    else:
        bad_ratio = 100*(bad_vlv.shape[0]/vlv_df.shape[0])
        long_tbad = bad_vlv['temp_diff'].describe()['mean']

    # estimate size of leak in terms of pct that valve is open
    if df_fit_nz is not None:
        est_leak = df_fit_nz[df_fit_nz['y_fitted'] <= long_tbad]['vlv_po'].max()
        if est_leak > popt[1] and bad_ratio > 5:
            passing_type['leak_grtr_xovr_fail'] = est_leak
    else:
        if bad_vlv is not None:
            est_leak = bad_vlv['temp_diff'].mean()
            if bad_ratio > 5 and est_leak > th_bad_vlv:
                passing_type['leak_grtr_threshold_fail'] = est_leak

    failure = [x in ['long_term_fail', 'leak_grtr_xovr_fail', 'leak_grtr_threshold_fail'] for x in passing_type.keys()]
    if any(failure):
        print_passing_mgs(row)
        folder = join(project_folder, bad_folder)
    else:
        folder = join(project_folder, good_folder)

    # categorized good and bad points
    vlv_df.loc[:, 'good_oper_cat'] = True
    if bad_vlv is not None:
        vlv_df.loc[bad_vlv.index, 'good_oper_cat'] = False

    _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=long_tc['25%'], long_tbad=long_tbad, df_fit=df_fit_nz, bad_ratio=bad_ratio, folder=folder)

    # TODO get a detailed report of the when valve is malfunctioning
    # lal = bad_vlv.groupby('same')
    # grps = list(lal.groups.keys())
    # bad_vlv.loc[lal.groups[grps[0]]]

    return passing_type

def _analyze_ahu(vlv_df, row, th_bad_vlv, th_time, project_folder):
    """
    Helper function to analyze AHU valves

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    row: Pandas series object with metadata for the specific valve

    th_bad_vlv: temperature difference from long term temperature difference to consider an operating point as malfunctioning

    th_time: length of time, in minutes, after the valve is closed to determine if 
            valve operating point is malfunctioning e.g. allow enough time for residue heat to
            dissipate from the coil.

    Returns
    -------
    None
    """
    if row['upstream_type'] != 'Mixed_Air_Temperature_Sensor':
        print('No upstream sensor data available for coil in AHU {} for site {}'.format(row['equip'], row['site']))
        #_make_tdiff_vs_vlvpo_plot(vlv_df, row, folder='./')
        passing_type = dict()
    else:
        passing_type = _analyze_vlv(vlv_df, row, th_bad_vlv, th_time, project_folder)

    return passing_type


def _analyze(metadata, fetch_resp, clean_func, analyze_func, th_bad_vlv, th_time, project_folder):
    """
    Hi level analyze function that runs through each valve queried to detect passing valves

    Parameters
    ----------
    metadata: metadata, i.e. view, for the valves that need to be analyzed

    fetch_resp : Mortar FetchResponse object

    clean_func: specific clean function for the valve in the equipment

    analyze_func: specific analyze function for the valve in the equipment

    th_bad_vlv: temperature difference from long term temperature difference to consider an operating point as malfunctioning

    th_time: length of time, in minutes, after the valve is closed to determine if 
            valve operating point is malfunctioning e.g. allow enough time for residue heat to
            dissipate from the coil.

    project_folder: name of path for the project and used to save the plots and csv data.

    Returns
    -------
    None
    """
    results = []
    # analyze valves
    for idx, row in metadata.iterrows():
        vlv_dat = dict(row)
        try:
            # clean data
            vlv_df = clean_func(fetch_resp, row)

            # analyze for passing valves
            passing_type = analyze_func(vlv_df, row, th_bad_vlv, th_time, project_folder)

        except:
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("******Error try to debug")
            print("{}: {}\n".format(exc_type, exc_value))
            print(''.join(traceback.format_tb(exc_traceback)))
            passing_type = dict()
            import pdb; pdb.set_trace()
            continue

        if passing_type is None:
            import pdb; pdb.set_trace()

        vlv_dat.update(passing_type)
        results.append(vlv_dat)

    final_df = pd.DataFrame.from_records(results)

    return final_df

def detect_passing_valves(eval_start_time, eval_end_time, window, th_bad_vlv, th_time, project_folder):
    """
    Main function that runs all the steps of the application

    Parameters
    ----------
    query: dictionary containing query, sites, and qualify response

    eval_start_time : start date and time in format (yyyy-mm-ddTHH:MM:SSZ) for the thermal
                      comfort evaluation period

    eval_end_time : end date and time in format (yyyy-mm-ddTHH:MM:SSZ) for the thermal
                    comfort evaluation period

    window : aggregation window, in minutes, to average the raw measurement data

    th_bad_vlv: temperature difference from long term temperature difference to consider an operating point as malfunctioning

    th_time: length of time, in minutes, after the valve is closed to determine if 
            valve operating point is malfunctioning e.g. allow enough time for residue heat to
            dissipate from the coil. If two values are defined, the 1st is for reheat coils and the 2nd for ahu coils.

    project_folder: name of path for the project and used to save the plots and csv data.

    Returns
    -------
    None
    """
    # declare user hidden parameters
    global long_term_fail   # number of minutes to trigger an long-term passing valve failure
    global shrt_term_fail   # number of minutes to trigger an intermitten passing valve failure
    global good_folder
    global bad_folder
    global air_flow_folder
    global csv_folder

    # define user hidden parameters
    long_term_fail = 5*60    # number of minutes to trigger an long-term passing valve failure
    shrt_term_fail = 60      # number of minutes to trigger an intermitten passing valve failure
    th_vlv_fail = 20         # equivalent percentage of valve open for determining failure.

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

    # split length of time for vav and ahus
    if isinstance(th_time, (list, tuple)):
        if len(th_time) == 2:
            th_time_vav = th_time[0]
            th_time_ahu = th_time[1]
        else:
            th_time_vav = th_time[0]
            th_time_ahu = th_time[0]
    else:
        th_time_vav = th_time
        th_time_ahu = th_time

    query = _query_and_qualify()
    fetch_resp = _fetch(query, eval_start_time, eval_end_time, window)

    # analyze VAV valves
    vav_metadata = fetch_resp['vav'].view('dnstream_ta')
    results_vav = _analyze(vav_metadata, fetch_resp['vav'], _clean_vav, _analyze_vlv, th_bad_vlv, th_time_vav, project_folder)

    # analyze AHU valves
    ahu_metadata = reformat_ahu_view(fetch_resp['ahu'])
    results_ahu = _analyze(ahu_metadata, fetch_resp['ahu'], _clean_ahu, _analyze_ahu, th_bad_vlv, th_time_ahu, project_folder)

    # clean report and save results
    final_df = pd.concat([results_vav, results_ahu])
    final_df = clean_final_report(final_df)
    final_df.to_csv(join(project_folder, "passing_valve_results" + ".csv"))


if __name__ == '__main__':
    # Disable options
    pd.options.mode.chained_assignment = None

    # define parameters
    eval_start_time  = "2018-07-01T00:00:00Z"
    eval_end_time    = "2018-12-31T23:59:00Z"
    window = 15
    th_bad_vlv = 10
    th_time = [12, 45]
    project_folder = './with_airflow_checks_year_end'

    # Run the app
    detect_passing_valves(eval_start_time, eval_end_time, window, th_bad_vlv, th_time, project_folder)