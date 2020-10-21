import pymortar
import sys
import pandas as pd
import numpy as np

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
    ?vav        rdf:type/rdfs:subClassOf? brick:VAV .
    ?vav        bf:isFedBy+                 ?ahu .
    ?vav_vlv    rdf:type                    ?vlv_type .
    ?ahu        bf:hasPoint                 ?ahu_supply .
    ?vav        bf:hasPoint                 ?vav_supply .
    ?ahu_supply rdf:type/rdfs:subClassOf*   brick:Supply_Air_Temperature_Sensor .
    ?vav_supply rdf:type/rdfs:subClassOf*   brick:Supply_Air_Temperature_Sensor .
    ?vav        bf:hasPoint                 ?vav_vlv .
    ?vav_vlv    rdf:type/rdfs:subClassOf*   brick:Valve_Command .
};"""

# find sites with these sensors and setpoints
qualify_resp = client.qualify([vav_query])
if qualify_resp.error != "":
    print("ERROR: ", qualify_resp.error)
    sys.exit(1)
elif len(qualify_resp.sites) == 0:
    print("NO SITES RETURNED")
    sys.exit(0)

print("running on {0} sites".format(len(qualify_resp.sites)))

# build the fetch request
request = pymortar.FetchRequest(
    sites=qualify_resp.sites,
    views=[
        pymortar.View(
            name="valves",
            definition=vav_query,
        ),
    ],
    dataFrames=[
        pymortar.DataFrame(
            name="valve",
            aggregation=pymortar.MEAN,
            window="15m",
            timeseries=[
                pymortar.Timeseries(
                    view="valves",
                    dataVars=["?vav_vlv"],
                )
            ]
        ),
        pymortar.DataFrame(
            name="vav_temp",
            aggregation=pymortar.MEAN,
            window="15m",
            timeseries=[
                pymortar.Timeseries(
                    view="valves",
                    dataVars=["?vav_supply"],
                )
            ]
        ),
        pymortar.DataFrame(
            name="ahu_temp",
            aggregation=pymortar.MEAN,
            window="15m",
            timeseries=[
                pymortar.Timeseries(
                    view="valves",
                    dataVars=["?ahu_supply"],
                )
            ]
        ),
    ],
    time=pymortar.TimeParams(
        start=eval_start_time,
        end=eval_end_time,
    )
)


# call the fetch api
fetch_resp = client.fetch(request)
print(fetch_resp)
print(fetch_resp.view('valves'))


# print the different types of valves in the data
#print(fetch_resp.view('valves').groupby(['vlv_subclass']).count())

def _clean(row):

    # combine data points in one dataframe
    vav_sa = fetch_resp['vav_temp'][row['vav_supply_uuid']]
    ahu_sa = fetch_resp['ahu_temp'][row['ahu_supply_uuid']]
    vlv_po = fetch_resp['valve'][row['vav_vlv_uuid']]

    vav_df = pd.concat([ahu_sa, vav_sa, vlv_po], axis=1)
    vav_df.columns = ['ahu_sa', 'vav_sa', 'vlv_po']

    # identify when valve is open
    vav_df['vlv_open'] = vav_df['vlv_po'] > 0

    # calculate temperature difference between ahu and vav supply air
    vav_df['temp_diff'] = vav_df['vav_sa'] - vav_df['ahu_sa']

    # drop na
    vav_df = vav_df.dropna()

    # drop values where vav supply air is less than ahu supply air
    vav_df = vav_df[vav_df['temp_diff'] >= 0]

    return vav_df


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

def get_fit_line(vav_df, x_col='vlv_po', y_col='temp_diff'):
    # fit the curve
    scaled_pos = scale_0to1(vav_df[x_col])
    scaled_t = scale_0to1(vav_df[y_col])
    popt, pcov = curve_fit(sigmoid, scaled_pos, scaled_t)

    # calculate fitted temp difference values
    est_k, est_x0 = popt
    y_fitted = rescale_fit(sigmoid(scaled_pos, est_k, est_x0), vav_df[y_col])
    y_fitted.name = 'y_fitted'

    # sort values
    df_fit = pd.concat([vav_df[x_col], y_fitted], axis=1)
    df_fit = df_fit.sort_values(by=x_col)

    return df_fit

def try_limit_dat_fit_model(vav_df, df_fraction):
    # calculate fit model
    nrows, ncols = vav_df.shape
    some_pts = np.random.choice(nrows, int(nrows*df_fraction))
    try:
        df_fit = get_fit_line(vav_df.iloc[some_pts])
    except RuntimeError:
        try:
            df_fit = get_fit_line(vav_df)
        except RuntimeError:
            print("No regression found")
            df_fit = None
    return df_fit

def calc_long_t_diff(vav_df, vlv_open=False):
    if vlv_open:
        # long-term average when valve is open
        df_vlv_close = vav_df[vav_df['vlv_open']]
    else:
        # long-term average when valve is closed
        df_vlv_close = vav_df[~vav_df['vlv_open']]

    long_t = df_vlv_close['temp_diff'].describe()

    return long_t

def _make_tdiff_vs_vlvpo_plot(vav_df, row, long_t=None, df_fit=None, folder='./'):
    # plot temperature difference vs valve position
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.scatter(x=vav_df['vlv_po'], y=vav_df['temp_diff'], alpha=1/3, s=10)
    ax.set_ylabel('Temperature difference')
    ax.set_xlabel('Valve pct opened')
    ax.set_title("Valve = {}\nVAV = {}".format(row['vav_vlv'], row['vav']), loc='left')

    if df_fit is not None:
        # add fit line
        ax.plot(df_fit['vlv_po'], df_fit['y_fitted'], '--', label='fitted', color='red')

    if long_t is not None:
        # add long-term temperature diff
        ax.axhline(y=long_t, color='green')

    plt_name = "{}-{}-{}".format(row['site'], row['vav'], row['vav_vlv'])
    plt.savefig(join(folder, plt_name + '.png'))

def return_exceedance(vav_df, long_t, th_time=45, window=15):
    # find datapoints that exceed long-term temperature difference
    min_ts = int(th_time/window)
    th_exceed = np.logical_and((vav_df['temp_diff'] >= long_t), ~(vav_df['vlv_open']))
    df_bad = vav_df[th_exceed]

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


valve_metadata = fetch_resp.view('valves')
import pdb; pdb.set_trace()

for idx, row in valve_metadata.iterrows():
    try:
        # clean data
        vav_df = _clean(row)

        if vav_df.shape[0] == 0:
            print("'{}' in site {} has no data! Skipping...".format(row['vav_vlv'], row['site']))
            continue

        # determine if valve datastream has open and closed data
        bool_type = vav_df['vlv_open'].value_counts().index

        bad_vlv_val = 5

        if len(bool_type) < 2:
            if bool_type[0]:
                # only open valve data
                long_to = calc_long_t_diff(vav_df, vlv_open=True)
                if long_to['50%'] < bad_vlv_val:
                    print("'{}' in site {} is open but seems to not cause an increase in air temperature\n".format(row['vav_vlv'], row['site']))
                    _make_tdiff_vs_vlvpo_plot(vav_df, row, long_t=long_to['50%'], folder='./bad_valves')
            else:
                # only closed valve data
                long_tc = calc_long_t_diff(vav_df)
                if long_tc['50%'] > bad_vlv_val:
                    print("Probable passing valve '{}' in site {}".format(row['vav_vlv'], row['site']))
                    _make_tdiff_vs_vlvpo_plot(vav_df, row, long_t=long_tc['50%'], folder='./bad_valves')
            continue

        # calculate long-term temp diff when valve is closed
        bad_klass = []
        long_tc = calc_long_t_diff(vav_df)
        long_to = calc_long_t_diff(vav_df, vlv_open=True)


        # make a simple comparison of between long-term open and long-term closed temp diff
        if (long_tc['mean'] + long_tc['std']) > long_to['mean']:
            print("Probable passing valve '{}' in site {}\n".format(row['vav_vlv'], row['site']))
            bad_klass.append(True)

        # assume a 0 deg difference at 0% open valve
        no_zeros_po = vav_df.copy()
        no_zeros_po.loc[no_zeros_po['vlv_po'] == 0, 'temp_diff'] = 0

        # # make a logit regression model based on a threshold value
        # # compare long-term average with actual temp diff
        # no_zeros_po['sig_diff'] = (no_zeros_po.loc[:, 'temp_diff'] > long_tc['50%']).astype(int)

        # df_fit_sig = get_fit_line(no_zeros_po, x_col='vlv_po', y_col='sig_diff')
        # df_fit_sig['y_fitted'] = rescale_fit(df_fit_sig['y_fitted'], no_zeros_po['temp_diff'])

        # est_lt_diff_sig = df_fit_sig[df_fit_sig['vlv_po'] == 0]['y_fitted'].mean()
        # bad_vlv = return_exceedance(vav_df, est_lt_diff_sig, th_time=45, window=15)

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
        th_ratio = 0.10
        bad_vlv = return_exceedance(vav_df, est_lt_diff_nz, th_time=45, window=15)

        if bad_vlv is None:
            bad_ratio = 0
        else:
            bad_ratio = bad_vlv.shape[0]/vav_df.shape[0]

        if bad_ratio > th_ratio:
            bad_klass.append(True)

        if len(bad_klass) > 0:
            folder = './bad_valves'
            print("Probable passing valve '{}' in site {}\n".format(row['vav_vlv'], row['site']))
            if len(bad_klass) > 1:
                print("{} percentage of time is leaking!".format(bad_ratio))
        else:
            folder = './good_valves'

        _make_tdiff_vs_vlvpo_plot(vav_df, row, long_t=long_tc['25%'], df_fit=df_fit_nz, folder=folder)

        # # get a detailed report of the when valve is malfunctioning
        # lal = bad_vlv.groupby('same')
        # grps = list(lal.groups.keys())
        # bad_vlv.loc[lal.groups[grps[0]]]


        # # logit fit with limited points
        # df_fit = try_limit_dat_fit_model(vav_df, df_fraction=1)
    except:
        print("Error try to debug")
        print(sys.exc_info()[0])
        import pdb; pdb.set_trace()
        continue