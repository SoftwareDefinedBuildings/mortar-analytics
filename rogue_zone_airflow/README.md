# Rogue Zone Airflow Detection

Finds extents of time where the measured airflow is lower than the corresponding setpoint for >= 2 hours.

This produces a CSV file called `rogue_zones.csv` when run. Each row is a possible incidence of a rogue zone:

- start and end of incident
- equipment name
- site
- length of incident
- average difference between airflow setpoint and sensor (in cfm)
