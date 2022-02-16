"""
Functions to analyze faults when detecting passing valves
in hot water hydronic systems.

@author Carlos Duarte <cduarte@berkeley.edu>
"""

def fault_sensor_inactivity(vlv_df, passing_type):
    """
    Check that sensors are taking measurements and not
    just reporting a constant number
    """
    pct_threshold = 1.5

    analysis_cols = ['upstream_ta', 'dnstream_ta', 'air_flow']
    df_cols = vlv_df.columns
    avail_cols = [ac for ac in analysis_cols if ac in df_cols]

    for col in avail_cols:
        col_dat = vlv_df.loc[:, col]
        col_max = max(col_dat)
        col_min = min(col_dat)

        # check if values are constant
        if col_max == col_min:
            passing_type["sensor_fault_CONSTANT_{}".format(col)] = (col_min, col_max)
            return vlv_df, passing_type

        col_dat = (col_dat-col_min)/(col_max-col_min)
        col_stats = abs(col_dat.diff(periods=-1).loc[vlv_df['cons_ts']]).describe()

        try:
            if round(col_stats["std"]*100,1) < pct_threshold:
                passing_type["sensor_fault_NORANGE_{}".format(col)] = [(key, col_stats[key]) for key in col_stats.keys()]
        except ValueError:
            import pdb; pdb.set_trace()

    return vlv_df, passing_type


def fault_sensor_out_of_range(vlv_df, passing_type):
    """
    Check that sensors are in the right value range
    """
    temp_max = 200
    temp_min = 20
    air_max  = 2500
    air_min  = -50
    pos_max  = 100
    pos_min  = 0

    analysis_cols = {
        'upstream_ta': [temp_max, temp_min],
        'dnstream_ta': [temp_max, temp_min],
        'air_flow': [air_max, air_min],
        'vlv_po': [pos_max, pos_min],
        }

    df_cols = vlv_df.columns
    for col in analysis_cols:
        if col in df_cols:
            col_dat = vlv_df.loc[:, col]
            val_max = analysis_cols[col][0]
            val_min = analysis_cols[col][1]

            vals_too_hi = sum(col_dat > val_max)/len(col_dat)
            vals_too_lo = sum(col_dat < val_min)/len(col_dat)

            if vals_too_hi > 0.05:
                passing_type["sensor_fault_HIVAL_{}".format(col)] = (vals_too_hi)

            if vals_too_lo > 0.05:
                passing_type["sensor_fault_LOVAL_{}".format(col)] = (vals_too_lo)

    return vlv_df, passing_type


