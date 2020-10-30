import pymortar
import sys
import pandas as pd
import numpy as np
import os

from os.path import join
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# define parameters
eval_start_time  = "2018-01-01T00:00:00Z"
eval_end_time    = "2018-06-30T00:00:00Z"

client = pymortar.Client()

# define query to return valves
# returns supply air temps from ahu and vav and vav valve
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

# build the fetch request
vav_request = pymortar.FetchRequest(
    sites=qualify_vav_resp.sites,
    views=[
        pymortar.View(
            name="dnstream_ta",
            definition=vav_query,
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


# build the fetch request
ahu_request = pymortar.FetchRequest(
    sites=ahu_sites,
    views=[
        pymortar.View(
            name="dnstream_ta",
            definition=ahu_sa_query,
        ),
        pymortar.View(
            name="upstream_ta",
            definition=ahu_ra_query,
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

def _clean_ahu_view(fetch_resp_ahu):
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

# call the fetch api for VAV data
fetch_resp_vav = client.fetch(vav_request)

print("-----Dataframe for VAV valves-----")
print(fetch_resp_vav)
print(fetch_resp_vav.view('dnstream_ta'))

# call the fetch api for AHU data
fetch_resp_ahu = client.fetch(ahu_request)
ahu_metadata = _clean_ahu_view(fetch_resp_ahu)

print("-----Dataframe for AHU valves-----")
print(fetch_resp_ahu)
print(ahu_metadata)


def _clean_vav(row):

    # combine data points in one dataframe
    vav_sa = fetch_resp_vav['dnstream_ta'][row['dnstream_ta_uuid']]
    ahu_sa = fetch_resp_vav['upstream_ta'][row['upstream_ta_uuid']]
    vlv_po = fetch_resp_vav['vlv'][row['vlv_uuid']]

    vav_df = pd.concat([ahu_sa, vav_sa, vlv_po], axis=1)
    vav_df.columns = ['upstream_ta', 'dnstream_ta', 'vlv_po']

    # identify when valve is open
    vav_df['vlv_open'] = vav_df['vlv_po'] > 0

    # calculate temperature difference between downstream and upstream air
    vav_df['temp_diff'] = vav_df['dnstream_ta'] - vav_df['upstream_ta']

    # drop na
    vav_df = vav_df.dropna()

    # drop values where vav supply air is less than ahu supply air
    vav_df = vav_df[vav_df['temp_diff'] >= 0]

    return vav_df

def _clean_ahu(row):
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

    # drop values where vav supply air is less than ahu supply air
    #ahu_df = ahu_df[ahu_df['temp_diff'] >= 0]

    return ahu_df


def scale_0to1(temp_diff):
    max_t = temp_diff.max()
    min_t = temp_diff.min()

    new_t = (temp_diff - min_t) / (max_t - min_t)

    return new_t

def rescale_fit(scaled_x, temp_diff):
    max_t = temp_diff.max()
    min_t = temp_diff.min()

    rescaled = min_t + scaled_x*(max_t - min_t)

    return rescaled

def sigmoid(x, k, x0):
    return 1.0 / (1 + np.exp(-k * (x - x0)))

def get_fit_line(vlv_df, x_col='vlv_po', y_col='temp_diff'):
    # fit the curve
    scaled_pos = scale_0to1(vlv_df[x_col])
    scaled_t = scale_0to1(vlv_df[y_col])
    popt, pcov = curve_fit(sigmoid, scaled_pos, scaled_t)

    # calculate fitted temp difference values
    est_k, est_x0 = popt
    y_fitted = rescale_fit(sigmoid(scaled_pos, est_k, est_x0), vlv_df[y_col])
    y_fitted.name = 'y_fitted'

    # sort values
    df_fit = pd.concat([vlv_df[x_col], y_fitted], axis=1)
    df_fit = df_fit.sort_values(by=x_col)

    return df_fit

def try_limit_dat_fit_model(vlv_df, df_fraction):
    # calculate fit model
    nrows, ncols = vlv_df.shape
    some_pts = np.random.choice(nrows, int(nrows*df_fraction))
    try:
        df_fit = get_fit_line(vlv_df.iloc[some_pts])
    except RuntimeError:
        try:
            df_fit = get_fit_line(vlv_df)
        except RuntimeError:
            print("No regression found")
            df_fit = None
    return df_fit

def calc_long_t_diff(vlv_df, vlv_open=False):
    if vlv_open:
        # long-term average when valve is open
        df_vlv_close = vlv_df[vlv_df['vlv_open']]
    else:
        # long-term average when valve is closed
        df_vlv_close = vlv_df[~vlv_df['vlv_open']]

    long_t = df_vlv_close['temp_diff'].describe()

    return long_t

def _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=None, long_tbad=None, df_fit=None, bad_ratio=None, folder='./'):
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

def return_exceedance(vlv_df, long_t, th_time=45, window=15):
    # find datapoints that exceed long-term temperature difference
    min_ts = int(th_time/window)
    th_exceed = np.logical_and((vlv_df['temp_diff'] >= long_t), ~(vlv_df['vlv_open']))
    df_bad = vlv_df[th_exceed]

    # only get consecutive timestamps datapoints
    ts = pd.Series(df_bad.index)
    ts_int = pd.Timedelta(window, unit='min')
    cons_ts = ((ts - ts.shift(-1)).abs() <= ts_int) | (ts.diff() <= ts_int)

    if (len(cons_ts) < min_ts) | ~(np.any(cons_ts)):
        return None

    #df_bad['cons_ts'] = np.array(cons_ts)
    df_bad.loc[:, 'cons_ts'] = np.array(cons_ts)
    df_bad.loc[:, 'same'] = df_bad['cons_ts'].astype(int).diff().ne(0).cumsum()
    #df_bad['same'] = df_bad['cons_ts'].astype(int).diff().ne(0).cumsum()

    df_cons_ts = df_bad[df_bad['cons_ts']]

    # subset by consecutive times that exceed th_time
    lal = df_cons_ts.groupby('same')
    grp_exceed = lal['same'].count()[lal['same'].count() >= min_ts].index

    exceeded = [x in grp_exceed for x in df_cons_ts['same']]
    bad_vlv = df_cons_ts[exceeded]

    return bad_vlv.drop(columns=['cons_ts'])

def check_folder_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def _analyze_vlv(vlv_df, row, bad_folder = './bad_valves', good_folder = './good_valves'):

    # check if holding folders exist
    check_folder_exist(bad_folder)
    check_folder_exist(good_folder)

    if vlv_df.shape[0] == 0:
        print("'{}' in site {} has no data! Skipping...".format(row['vlv'], row['site']))
        return

    # determine if valve datastream has open and closed data
    bool_type = vlv_df['vlv_open'].value_counts().index

    bad_vlv_val = 5

    if len(bool_type) < 2:
        if bool_type[0]:
            # only open valve data
            long_to = calc_long_t_diff(vlv_df, vlv_open=True)
            if long_to['50%'] < bad_vlv_val:
                print("'{}' in site {} is open but seems to not cause an increase in air temperature\n".format(row['vlv'], row['site']))
                _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=long_to['50%'], folder=bad_folder)
        else:
            # only closed valve data
            long_tc = calc_long_t_diff(vlv_df)
            if long_tc['50%'] > bad_vlv_val:
                print("Probable passing valve '{}' in site {}".format(row['vlv'], row['site']))
                _make_tdiff_vs_vlvpo_plot(vlv_df, row, long_t=long_tc['50%'], folder=bad_folder)
        return

    # calculate long-term temp diff when valve is closed
    bad_klass = []
    long_tc = calc_long_t_diff(vlv_df)
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
        df_fit_nz = get_fit_line(no_zeros_po)
    except RuntimeError:
        df_fit_nz = None

    # determine estimated long-term difference
    if df_fit_nz is not None:
        est_lt_diff_nz = df_fit_nz[df_fit_nz['vlv_po'] == 0]['y_fitted'].mean()
    else:
        est_lt_diff_nz = long_tc['25%']

    # calculate bad valve instances vs overall dataframe
    th_ratio = 20
    bad_vlv = return_exceedance(vlv_df, est_lt_diff_nz, th_time=45, window=15)

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

def _analyze_ahu(vlv_df, row):
    import pdb; pdb.set_trace()

    if row['upstream_type'] != 'Mixed_Air_Temperature_Sensor':
        print('No upstream sensor data available for coil in AHU {} for site {}'.format(row['equip'], row['site']))
        #_make_tdiff_vs_vlvpo_plot(vlv_df, row, folder='./')
    else:
        _analyze_vlv(vlv_df, row)


def analyze(metadata, clean_func, analyze_func):
    # analyze valves
    for idx, row in metadata.iterrows():
        try:
            # clean data
            vlv_df = clean_func(row)

            # analyze for passing valves
            analyze_func(vlv_df, row)

        except:
            print("Error try to debug")
            print(sys.exc_info()[0])
            import pdb; pdb.set_trace()
            continue

vav_metadata = fetch_resp_vav.view('dnstream_ta')

# analyze VAV valves
analyze(vav_metadata, _clean_vav, _analyze_vlv)

# analyze AHU valves
analyze(ahu_metadata, _clean_ahu, _analyze_ahu)
