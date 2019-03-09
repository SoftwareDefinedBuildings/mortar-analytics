# Detecting Zones that Heat and Cool

Finds all the zones that both heat and cool in the same hour. 

This produces a CSV file called `simultaneous_heat_cool_zones.csv` when run. Each row is contains the following information:
* time: the hour in which heating and cooling occured
* site
* zone
* room
* heat_percent: Percentage of time in that hour for which the zone was heated
* cool_percent: Percentage of time in that hour for which the zone was cooled
* min_hsp: Minimum heating setpoint during that hour
* min_csp: Minimum cooling setpoint during that hour
* max_hsp: Maximum heating setpoint during that hour
* max_csp: Maximum cooling setpoint during that hour