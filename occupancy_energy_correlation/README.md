# Building occupancy and energy consumption correlation

This app calculates the absolute and percent building energy consumption when the building is occupied. Additionally, it creates a plot for each building with its energy consumption data and occupancy data overlapping.


# Run the app
Edit the parameters in config.json,
- results_folder: name of the results folder (all plots will be saved here). Defaults to "results".
- sites: Name of site. "" defaults to all sites in Green Button Meter. 
- time: start and end determine the range of data to query. Note: Use the format given in config.json to edit time.
- save_data: boolean variable to indicate whether to save data after querying mortar.

Execute application

```
python app.py
```

The app will print out the results in the terminal and save the plots in the results folder.
