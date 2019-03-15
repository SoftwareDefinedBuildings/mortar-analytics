# Detecting AHUs whose Heating Valve and Cooling Valve are open simultaneously

This app detects AHUs whose heating and cooling valves were both open at the same time. \

## Run this app
`python app.py [-st <start_time> -et <end_time> -filename <filename>]`

* `start_time`: start time for analysis in yyyy-mm-ddThh:mm:ss format (default: 2017-06-21T00:00:00)
* `end_time`: end time for analysis in yyyy-mm-ddThh:mm:ss format (default: 2017-07-01T00:00:00)
* `filename`: name of the csv file to store result of analysis (default: simultaneous_heat_cool_ahu.csv)

## Output

This produces a CSV file called `simultaneous_heat_cool_ahu.csv` (default) when run. Each row is contains the following information:
* time: the time in which both the heating and cooling valves were open
* site
* ahu