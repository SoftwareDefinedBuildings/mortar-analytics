# Detecting Zones that Heat and Cool in the same time interval

This app detects zones that were both heated and cooled in the same time interval (default interval of 1 hour). \
Heating and cooling in the same interval does not necessarily point towards malfunctioning equipment in the zone, but it does suggest that there are possible changes that can be implemented to improve the operating efficiency. 

## Run this app
`python app.py [-st <start_time> -et <end_time> -time_interval <time_interval> -filename <filename>]`

* `start_time`: start time for analysis in yyyy-mm-ddThh:mm:ss format (default: 2018-12-10T00:00:00)
* `end_time`: end time for analysis in yyyy-mm-ddThh:mm:ss format (default: 2019-01-01T00:00:00)
* `time_interval`: length of time interval (in minutes) when you want to check if a zone is both heating and cooling (default: 60)
* `filename`: name of the csv file to store result of analysis (default: heat_and_cool_same_period.csv)

## Output

This produces a CSV file called `heat_and_cool_same_period.csv` (default) when run. Each row is contains the following information:
* time: the start time of the period in which heating and cooling occured
* site
* zone
* room
* heat_percent: Percentage of time in that time interval for which the zone was heated
* cool_percent: Percentage of time in that time interval for which the zone was cooled
* min_hsp: Minimum heating setpoint during that time interval
* min_csp: Minimum cooling setpoint during that time interval
* max_hsp: Maximum heating setpoint during that time interval
* max_csp: Maximum cooling setpoint during that time interval
