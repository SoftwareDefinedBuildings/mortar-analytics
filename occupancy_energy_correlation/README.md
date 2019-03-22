# Building occupancy and energy correlation

This app calculates the absolute and percent building energy consumption when the building is unoccupied.

# Run the app
Edit the parameters in config.json,
- sites: Name of site. "" defaults to all sites in Green Button Meter. 
- time: start and end determine the range of data to query. Note: Use the format given in config.json to edit time.
- aggregation: Available aggregations for pymortar include - pymortar.MEAN, pymortar.MAX, pymortar.MIN, pymortar.COUNT, pymortar.SUM, pymortar.RAW (the temporal window parameter is ignored)
- window: specifies the interval of data.
- save_data: boolean variable to indicate whether to save data after querying mortar.

Execute the command,
python app.py

The app will print out the results in the terminal.
