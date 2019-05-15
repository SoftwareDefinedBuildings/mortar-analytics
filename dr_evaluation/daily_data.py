from .calc_price import calc_price, power_15min_to_hourly_energy

def get_daily_data(site, actual, baseline):
    start_time = actual.index[0]
    end_time = actual.index[-1]
    event = start_time.date()

    # Calculate costs
    actual_cost = calc_price(actual, site, start_time, end_time)
    baseline_cost = calc_price(baseline, site, start_time, end_time)

    # Calculate energy in kwH
    actual_energy = power_15min_to_hourly_energy(actual) / 1000
    baseline_energy = power_15min_to_hourly_energy(baseline) / 1000

    return {
        'site': site,
        'date': event,
        'actual_energy': sum(actual_energy),
        'baseline_energy': sum(baseline_energy),
        'energy_savings_day': sum(baseline_energy) - sum(actual_energy),
        'actual_energy_during_event': sum(actual_energy[14:18]),
        'baseline_energy_during_event': sum(baseline_energy[14:18]),
        'energy_savings_event': sum(baseline_energy[14:18]) - sum(actual_energy[14:18]),
        'actual_cost': actual_cost,
        'baseline_cost': baseline_cost,
        'savings': baseline_cost - actual_cost
    }