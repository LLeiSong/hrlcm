## Image preprocessing

### Introduction
This document concisely shows the steps to preprocess images for this project. 
The main tool is python package [sentinelPot](https://github.com/LLeiSong/sentinelPot).
More details could be found within the README of this repo. 
All steps are packed into independent scripts that can be called directly using python.
The user can surely customize their own preprocessing scripts based on the document of [sentinelPot](https://github.com/LLeiSong/sentinelPot).

### Configures
The main tedious part of preprocessing is to set configure yaml files.

- `config_main_template.yaml` is the yaml for sentinel-1 & 2. The user could set each argument based on the template. Then the scripts are ready to go.
- `config_plt_template.yaml` is the yaml for Planet NICFI query.

### Sentinel-1
Here are a few scripts under folder `scripts/sentinel` for sentinel-1 preprocessing:

- `s12_peps_query.py` is a script to query raw sentinel-1 imagery from [peps](https://peps.cnes.fr) server.
- `s1_preprocess.py` is a script to run recommended preprocessing steps for sentinel-1 imagery locally.
- `s1_harmonic_regression.py` is a script to calculate harmonic regression coefficients using Lasso algorithm. Before regression, a guided image filter is used to remove speckles further.

### Sentinel-2
Here are a few scripts under folder `scripts/sentinel` for sentinel-2 preprocessing:

- `s12_peps_query.py` is a script to query raw sentinel-2 imagery from [peps](https://peps.cnes.fr) server.
- `s2_preprocess.py` is a script to do atmospheric correction and cloud/shadow detection using MAJA processor installed on [peps](https://peps.cnes.fr) server and then downloaded all products into local machine.
- `s2_preprocess_steps.py` does the same as `s1_preprocess.py`, but has each steps separately. So the user could control each step flexibly. Sometimes it is convenient because [peps](https://peps.cnes.fr) server might be unstable occasionally. 
- `s2_maja_download.py` is a script to just download MAJA processed sentinel-2 tiles from [peps](https://peps.cnes.fr) server. It only can be used if you request MAJA processing on [peps](https://peps.cnes.fr) server before.
- `s2_wasp.py` is a script to run WASP docker to make temporal syntheses of MAJA processed sentinel-2 tiles.

### PlanetScope
We are using [Planet’s high-resolution, analysis-ready mosaics of the world’s tropics](https://www.planet.com/nicfi/). 
For archive years, there are only two seasons for each year.
Please find more details in its [user guides](https://assets.planet.com/docs/NICFI_UserGuidesFAQ.pdf).
The main script is `scripts/nicfi/nicfi_query.py`, which is written based upon Python package [planet](https://github.com/planetlabs/planet-client-python) and its [API document](https://planetlabs.github.io/planet-client-python/index.html).
