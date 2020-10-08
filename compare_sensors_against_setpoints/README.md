# Compare Sensor Measurements against Setpoints

Compare sensor measurements against their respective setpoints. The app can retrieve sensors where their measurements are above or below a setpoint, in between both the minimum and maximum setpoint values of the setpoint, or measurement exceeds either the minimum or maximum setpoint values.

The app produces a CSV file called `<sensor>_measure_vs_setpoint_<type of analysis>.csv` when run where '<sensor>' states the sensor type and '<analysis>' states the type of analysis performed. Each row contains the following information: 

- site name
- equipment name
- number of hours that sensor measurement meets the selected criteria
- start date and time that sensor measurement meets the selected criteria
- end date and time that sensor measurement meets the selected criteria