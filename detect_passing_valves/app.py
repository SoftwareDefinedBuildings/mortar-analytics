import pymortar
import sys
import pandas as pd
import numpy as np
import os

from os.path import join
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

        # drop values where there is no air flow
        vav_df = vav_df.loc[vav_df['air_flow'] > 0]
    else:
        vav_df = pd.concat([ahu_sa, vav_sa, vlv_po], axis=1)
        vav_df.columns = ['upstream_ta', 'dnstream_ta', 'vlv_po']

        # drop values outside occupancy hours
        vav_df = occupied_hours_subset(vav_df, occ_str=6, occ_end=18, wkend_str=5)

    # identify when valve is open
    vav_df['vlv_open'] = vav_df['vlv_po'] > 0

    # calculate temperature difference between downstream and upstream air
    vav_df['temp_diff'] = vav_df['dnstream_ta'] - vav_df['upstream_ta']

    # drop na
    vav_df = vav_df.dropna()

    # drop values where vav supply air is less than ahu supply air
    vav_df = vav_df[vav_df['temp_diff'] >= 0]

    return vav_df

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
               Please check or press 'c' to continue using the first sensor.".format(row['equip'], row['site']))
        print(flow_meta)
        import pdb; pdb.set_trace()
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
    ahu_df = ahu_df.dropna()

    # drop values outside occupancy hours
    ahu_df = occupied_hours_subset(ahu_df, occ_str=6, occ_end=18, wkend_str=5)

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


def rescale_fit(scaled_vals, vals):
    """
    Rescale values of pandas series that are 0 to 1 to match the interval
    of another pandas series object values

    Parameters
    ----------
    scaled_vals: Pandas series object with values scaled from 0 to 1 and needs to be unscaled

    vals: Pandas series object or Pandas dataframe colum with unnormalized values.
          This is used to extract max and min to unscaled the scaled_vals.

    Returns
    -------
    unscaled_vals: Pandas series object of unscaled values
    """
    max_val = vals.max()
    min_val = vals.min()

    unscaled_vals = min_val + scaled_vals*(max_val - min_val)

    return unscaled_vals


def sigmoid(x, k, x0):
    """
    Sigmoid function curve to do a logistic model

    Parameters
    ----------
    x: independent variable
    k: slope of the sigmoid function
    x0: midpoint of the sigmoid function

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
    """

    # fit the curve
    scaled_pos = scale_0to1(df[x_col])
    scaled_t = scale_0to1(df[y_col])
    popt, pcov = curve_fit(sigmoid, scaled_pos, scaled_t)

    # calculate fitted temp difference values
    est_k, est_x0 = popt
    y_fitted = rescale_fit(sigmoid(scaled_pos, est_k, est_x0), df[y_col])
    y_fitted.name = 'y_fitted'

    # sort values
    df_fit = pd.concat([df[x_col], y_fitted], axis=1)
    df_fit = df_fit.sort_values(by=x_col)

    return df_fit

def try_limit_dat_fit_model(vlv_df, df_fraction):
    # calculate fit model
    nrows, ncols = vlv_df.shape
    some_pts = np.random.choice(nrows, int(nrows*df_fraction))
    try:
        df_fit = build_logistic_model(vlv_df.iloc[some_pts])
    except RuntimeError:
        try:
            df_fit = build_logistic_model(vlv_df)
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

def calc_long_t_diff(vlv_df, vlv_open=False, row=None):
    """
    Calculate statistic on difference between down- and up-
    stream temperatures to determine the long term temperature difference.

    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    vlv_open: boolean define if the statistics are performed on data that has
              valve open (True) or closed (False)

    Returns
    -------
    long_t: dictionary object with statisitics of temperature difference
    """
    if vlv_open:
        # long-term average when valve is open
        df_vlv_close = vlv_df[vlv_df['vlv_open']]
    else:
        # long-term average when valve is closed and only values after th_time minutes
        # after valve has closed is included in average

        #df_vlv_close = vlv_df[~vlv_df['vlv_open']]
        df_vlv_close = return_delayed_df(vlv_df, th_time=25, window=15)
        if row is not None:
            check_folder_exist("./csv_data")
            _name = "{}-{}-{}_dat".format(row['site'], row['equip'], row['vlv'])
            df_vlv_close.to_csv(join("./csv_data", _name + '.csv'))

        df_vlv_close = df_vlv_close[np.logical_and(df_vlv_close['cons_ts_vlv_c'], df_vlv_close['steady'])]

    long_t = df_vlv_close['temp_diff'].describe()

    return long_t

def return_delayed_df(df_subset, th_time, window):
    """
    Return dataframe with row values that are X time after a changed state
    """

    min_ts = int(th_time/window) + (th_time % window > 0)
    min_tst = pd.Timedelta(th_time, unit='min')

    # only get consecutive timestamps datapoints
    ts = pd.Series(df_subset.index)
    ts_int = pd.Timedelta(window, unit='min')
    cons_ts = ((ts - ts.shift(-1)).abs() <= ts_int) | (ts.diff() <= ts_int)

    if (len(cons_ts) < min_ts) | ~(np.any(cons_ts)):
        return None

    df_subset['cons_ts'] = np.array(cons_ts)
    df_subset['cons_ts_vlv_c'] = np.logical_and(~df_subset['vlv_open'], df_subset['cons_ts'])
    df_subset['same'] = df_subset['cons_ts_vlv_c'].astype(int).diff().ne(0).cumsum()

    df_cons_ts = df_subset.copy()

    # subset by consecutive times that exceed th_time
    lal = df_cons_ts.groupby('same')

    steady = []
    for grp in lal.groups.keys():
        for ts in lal.groups[grp]:
            init_ts = lal.groups[grp][0]
            steady.append(init_ts+min_tst < ts)

    df_cons_ts['steady'] = np.array(steady)

    return df_cons_ts


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
    # plot temperature difference vs valve position
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.set_ylabel('Temperature difference [°F]')
    ax.set_xlabel('Valve opened [%]')
    ax.set_title("Valve = {}\nEquip. = {}".format(row['vlv'], row['equip']), loc='left')

    if 'color' in vlv_df.columns:
        ax.scatter(x=vlv_df['vlv_po'], y=vlv_df['temp_diff'], color = vlv_df['color'], alpha=1/3, s=10)
    else:
        ax.scatter(x=vlv_df['vlv_po'], y=vlv_df['temp_diff'], color = '#005ab3', alpha=1/3, s=10)

    if df_fit is not None:
        # add fit line
        ax.plot(df_fit['vlv_po'], df_fit['y_fitted'], '--', label='fitted', color='#5900b3')

    if long_t is not None:
        # add long-term temperature diff
        ax.axhline(y=long_t, color='#00b3b3')

    if long_tbad is not None:
        ax.axhline(y=long_tbad, color='#ff8cc6')

    if bad_ratio is not None:
        # add ratio where presumably passing valve
        y_max = vlv_df['temp_diff'].max()
        ax.text(.2, 0.95*y_max, "Bad ratio={:.1f}%".format(bad_ratio))

    plt_name = "{}-{}-{}".format(row['site'], row['equip'], row['vlv'])
    plt.savefig(join(folder, plt_name + '.png'))
    plt.close()


def find_bad_vlv_operation(vlv_df, long_t, th_time=45, window=15):
    """


    Parameters
    ----------
    vav_df: Pandas dataframe with valve timeseries data

    long_t: long-term temperature difference between down and up air streams when valve is 
            commanded close for correct operation

    th_time: length of time, in minutes, after the valve is closed to determine if 
            valve operating point is malfunctioning e.g. allow enough time for residue heat to
            dissipate from the coil.

    window : aggregation window, in minutes, to average the raw measurement data

    Returns
    -------
    bad_vlv: pandas dataframe object with the time intervals that the valve is malfunctioning
    """

    # find datapoints that exceed long-term temperature difference
    min_ts = int(th_time/window) + (th_time % window > 0)
    th_exceed = np.logical_and((vlv_df['temp_diff'] >= long_t), ~(vlv_df['vlv_open']))
    df_bad = vlv_df[th_exceed]

    # only get consecutive timestamps datapoints
    ts = pd.Series(df_bad.index)
    ts_int = pd.Timedelta(window, unit='min')
    cons_ts = ((ts - ts.shift(-1)).abs() <= ts_int) | (ts.diff() <= ts_int)

    if (len(cons_ts) < min_ts) | ~(np.any(cons_ts)):
        return None

    #df_bad['cons_ts'] = np.array(cons_ts)
    df_bad['cons_ts'] = np.array(cons_ts)
    df_bad['same'] = df_bad['cons_ts'].astype(int).diff().ne(0).cumsum()
    #df_bad['same'] = df_bad['cons_ts'].astype(int).diff().ne(0).cumsum()

    df_cons_ts = df_bad[df_bad['cons_ts']]

    # subset by consecutive times that exceed th_time
    lal = df_cons_ts.groupby('same')
    grp_exceed = lal['same'].count()[lal['same'].count() >= min_ts].index

    exceeded = [x in grp_exceed for x in df_cons_ts['same']]
    bad_vlv = df_cons_ts[exceeded]

    return bad_vlv.drop(columns=['cons_ts'])


def _analyze_vlv(vlv_df, row, th_bad_vlv=5, th_time=45, good_folder='./good_valves', bad_folder='./bad_valves'):
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

    good_folder: name of path showing the folder to save the plots of the correct operating valves

    bad_folder: name of path showing the folder to save the plots of the malfunction valves

    Returns
    -------
    None
    """

    # check if holding folders exist
    check_folder_exist(bad_folder)
    check_folder_exist(good_folder)

    # container for holding types of faults
    bad_klass = []

    if vlv_df.shape[0] == 0:
        print("'{}' in site {} has no data! Skipping...".format(row['vlv'], row['site']))
        return

    # determine if valve datastream has open and closed data
    bool_type = vlv_df['vlv_open'].value_counts().index

    if len(bool_type) < 2:
        if bool_type[0]:
            # only open valve data
            long_to = calc_long_t_diff(vlv_df, vlv_open=True)
            if long_to['50%'] < th_bad_vlv:
                print("'{}' in site {} is open but seems to not cause an increase in air temperature\n".format(row['vlv'], row['site']))
                _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=long_to['50%'], folder=bad_folder)
        else:
            # only closed valve data
            long_tc = calc_long_t_diff(vlv_df)
            if long_tc['50%'] > th_bad_vlv:
                print("Probable passing valve '{}' in site {}".format(row['vlv'], row['site']))
                _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=long_tc['50%'], folder=bad_folder)
        return

    # calculate long-term temp diff when valve is closed
    long_tc = calc_long_t_diff(vlv_df, row=row)
    long_to = calc_long_t_diff(vlv_df, vlv_open=True)


    # make a simple comparison of between long-term open and long-term closed temp diff
    if (long_tc['mean'] + long_tc['std']) > long_to['mean']:
        print("Probable passing valve '{}' in site {}\n".format(row['vlv'], row['site']))
        bad_klass.append(True)

    # assume a 0 deg difference at 0% open valve
    no_zeros_po = vlv_df.copy()
    no_zeros_po.loc[no_zeros_po['vlv_po'] == 0, 'temp_diff'] = 0

    # make a logit regression model assuming that closed valves make a zero temp difference
    try:
        df_fit_nz = build_logistic_model(no_zeros_po)
    except RuntimeError:
        df_fit_nz = None

    # determine estimated long-term difference
    if df_fit_nz is not None:
        est_lt_diff_nz = df_fit_nz[df_fit_nz['vlv_po'] == 0]['y_fitted'].mean()
    else:
        est_lt_diff_nz = long_tc['25%']

    # calculate bad valve instances vs overall dataframe
    th_ratio = 20
    bad_vlv = find_bad_vlv_operation(vlv_df, est_lt_diff_nz, th_time, window)

    if bad_vlv is None:
        bad_ratio = 0
        long_tbad = long_tc['mean']
    else:
        bad_ratio = 100*(bad_vlv.shape[0]/vlv_df.shape[0])
        long_tbad = bad_vlv['temp_diff'].describe()['mean']

    if df_fit_nz is not None:
        est_leak = df_fit_nz[df_fit_nz['y_fitted'] <= long_tbad]['vlv_po'].max()
    else:
        est_leak = bad_ratio

    if est_leak > th_ratio:
        bad_klass.append(True)

    if len(bad_klass) > 0:
        folder = bad_folder
        if bad_ratio > 5:
            print("Probable passing valve '{}' in site {}\n".format(row['vlv'], row['site']))
            if len(bad_klass) > 1:
                print("{} percentage of time is leaking!".format(bad_ratio))
        else:
            folder = good_folder
    else:
        folder = good_folder

    if bad_vlv is not None:
        # colorize good and bad points
        vlv_df['color'] = '#5ab300'
        vlv_df.loc[bad_vlv.index, 'color'] = '#b3005a'

    _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=long_tc['25%'], long_tbad=long_tbad, df_fit=df_fit_nz, bad_ratio=bad_ratio, folder=folder)

    # TODO get a detailed report of the when valve is malfunctioning
    # lal = bad_vlv.groupby('same')
    # grps = list(lal.groups.keys())
    # bad_vlv.loc[lal.groups[grps[0]]]

def _analyze_ahu(vlv_df, row, th_bad_vlv, th_time, good_folder, bad_folder):
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

    good_folder: name of path showing the folder to save the plots of the correct operating valves

    bad_folder: name of path showing the folder to save the plots of the malfunction valves

    Returns
    -------
    None
    """

    if row['upstream_type'] != 'Mixed_Air_Temperature_Sensor':
        print('No upstream sensor data available for coil in AHU {} for site {}'.format(row['equip'], row['site']))
        #_make_tdiff_vs_vlvpo_plot(vlv_df, row, folder='./')
    else:
        _analyze_vlv(vlv_df, row, th_bad_vlv, th_time, good_folder, bad_folder)


def _analyze(metadata, fetch_resp, clean_func, analyze_func, th_bad_vlv, th_time, good_folder, bad_folder):
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

    good_folder: name of path showing the folder to save the plots of the correct operating valves

    bad_folder: name of path showing the folder to save the plots of the malfunction valves

    Returns
    -------
    None
    """
    # analyze valves
    for idx, row in metadata.iterrows():
        try:
            # clean data
            vlv_df = clean_func(fetch_resp, row)

            # analyze for passing valves
            analyze_func(vlv_df, row, th_bad_vlv, th_time, good_folder, bad_folder)

        except:
            print("Error try to debug")
            print(sys.exc_info()[0])
            import pdb; pdb.set_trace()
            continue

def detect_passing_valves(eval_start_time, eval_end_time, window, th_bad_vlv, th_time, good_folder, bad_folder):
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
            dissipate from the coil.

    good_folder: name of path showing the folder to save the plots of the correct operating valves

    bad_folder: name of path showing the folder to save the plots of the malfunction valves



    Returns
    -------
    None
    """
    query = _query_and_qualify()
    fetch_resp = _fetch(query, eval_start_time, eval_end_time, window)

    # analyze VAV valves
    vav_metadata = fetch_resp['vav'].view('dnstream_ta')
    _analyze(vav_metadata, fetch_resp['vav'], _clean_vav, _analyze_vlv, th_bad_vlv, th_time, good_folder, bad_folder)

    # analyze AHU valves
    ahu_metadata = reformat_ahu_view(fetch_resp['ahu'])
    _analyze(ahu_metadata, fetch_resp['ahu'], _clean_ahu, _analyze_ahu, th_bad_vlv, th_time, good_folder, bad_folder)


if __name__ == '__main__':
    # define parameters
    eval_start_time  = "2018-01-01T00:00:00Z"
    eval_end_time    = "2018-06-30T00:00:00Z"
    window = 15
    th_bad_vlv = 5
    th_time = 45
    good_folder = './good_valves'
    bad_folder = './bad_valves'

    # Run the app
    detect_passing_valves(eval_start_time, eval_end_time, window, th_bad_vlv, th_time, good_folder, bad_folder)