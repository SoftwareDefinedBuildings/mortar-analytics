# Detect passing valves in VAV terminals

This app detects valves that do not close all the way also known as passing valves.

This app produces a CSV file called `passing_valves.csv` when run. Each row is a possible incidence of a passing valve. The CSV file contains the following columns:

- site
- valve name
- start of incident
- end of incident
- expected temperature difference
- actual temperature difference