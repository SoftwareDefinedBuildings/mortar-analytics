# Mortar Analytics Library

Welcome to the Mortar Analytics Library!
This is an **open source**, **community-driven** library of portable building analytics intended to be used with the [Mortar](https://mortardata.org/) data platform.

For a complete description of the Mortar project, please visit [mortardata.org](https://mortardata.org/)

## Have Questions?

- You can [create an issue](https://github.com/SoftwareDefinedBuildings/mortar-analytics/issues/new)
- You can [subscribe](https://lists.eecs.berkeley.edu/sympa/subscribe/mortar-users) to the mailing list: `mortar-users [ AT ] lists [  DOT ] eecs [  DOT ] berkeley [  DOT ] edu`

## How to Use

1. Follow the [Quick Start](https://mortardata.org/docs/quick-start/) guide. This boils down to the following:
    - make an account on [mortardata.org](https://mortardata.org/)
    - clone this repository
    - install the PyMortar library with Python >= 3.5:
        `pip install pymortar`

2. `cd` into the directory within `mortar-analytics` of the application you want to run

    ```
    cd mortar-analytics/meter_data_example
    ```

3. (optional, but recommended) Install the app's dependencies using the requirements.txt in a virtual environment

    ```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4. Execute the application

    ```
    python -i app.py
    ```

    You can also execute applications inside a Jupyter notebook context, which will allow you to browse generated visualizations in the browser along with the code.
    The Docker container in the Quick Start (step 1) will create this environment for you

## How to Contribute

See [CONTRIBUTING](https://github.com/SoftwareDefinedBuildings/mortar-analytics/blob/master/CONTRIBUTING.md)

---

Mortar is a research project from UC Berkeley.
