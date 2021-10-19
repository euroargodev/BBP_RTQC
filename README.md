# Introduction
This repository contains a new set of real-time quality-control (RTQC) test for particulate optical backscattering (BBP) data collected by BGC-Argo floats.

# A bit of history
A draft set of BBP RTQC tests and their impact on the GDAC data were presented at the ADMT21 meeting in 2020: https://drive.google.com/file/d/1Kkof40Cj5zWuVHQiRpudBRybXW6hJ4VQ/view?usp=sharing

A group of interested people met after ADMT21 to discuss the proposed tests further. At that time, I updated the presentation with the feedback/notes/decisions taken: https://drive.google.com/file/d/1JxdmBCS2MhG6qzmHwpmTCb-D5D3uX3qC/view?usp=sharing

The current version (30-Sep-2021) of the tests has not been presented to the community yet.

# Set up and installation
To work with these examples you first need to clone this repository (e.g., click [`here`](https://github.com/euroargodev/BBP_RTQC/archive/refs/heads/main.zip).

The file `environment.yml` allows you to recreate the `conda` environment needed for the notebooks to work. If you do not have `conda` installed, follow the instructions here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/.

To create the `BBP_RTQC` environment use the command `conda env create -f environment.yml`. Then the newly created conda environment needs to be activated using `conda activate BBP_RTQC`.

# Code
The code is written in Python 3 as scripts or Jupyter Notebooks.

### Test functions
[`BBP_RTQC.py`](https://github.com/euroargodev/BBP_RTQC/blob/main/BBP_RTQC.py): This script contains all the proposed tests, besides other functions that are needed to QC the data. It is loaded as a module by various notebooks.

### Global variables
[`BBP_RTQC_global_vars.py`](https://github.com/euroargodev/BBP_RTQC/blob/main/BBP_RTQC_global_vars.py): This script contains all global variables needed by the different notebooks. It is loaded by various notebooks.

[`BBP_RTQC_paths.py`](https://github.com/euroargodev/BBP_RTQC/blob/main/BBP_RTQC_paths.py): This script contains the paths to the data, the working directory, as well as the directories where the pickled results and the plots are stored. The paths currently stored here are those for my machine, so you'll have to update these to the ones in yours.

### How to run tests on GDAC profiles
[`run_BBP_RTQC.ipynb`](https://github.com/euroargodev/BBP_RTQC/blob/main/run_BBP_RTQC.ipynb): This is the main notebook used to apply the BBP RTQC tests. After importing the main modules used, it asserts all tests, and creates a list of DAC/WMO to be processed. It then removes any existing pickled and plot files. Finally, it runs the tests on each float in parallel (using 7 core CPUs, i.e., `n_jobs`: you will need to update this argument based on your machine).
There are also a number of commented cells that can be used to run the tests sequentially, while being able to read the output (set `VERBOSE = True`).

### How to plot and synthesize results
[`bgc_argo_RTQC_plot_results.ipynb`](https://github.com/euroargodev/BBP_RTQC/blob/main/bgc_argo_RTQC_plot_results.ipynb): This notebook is used to present a summary of the results of each test on the GDAC data. After reading all the pickled files, it extracts statistics on each RTQC test, presents them in a tabular format and generates plots for each test. It also creates directories for the plots related to each test.

### Asserting BBP RTQC tests
To ensure reproducibility, a set of example inputs and expected outputs is stored in the file [`BBP_RTQC_example_tests.json`](https://github.com/euroargodev/BBP_RTQC/blob/main/BBP_RTQC_example_tests.json).
This file is generated using [`prep_json.ipynb`](https://github.com/euroargodev/BBP_RTQC/blob/main/prep_json.ipynb) and is based on a set of profiles that may not be available anymore in the future (e.g., when R files become D files).

The notebooks 
[`check_tests.ipynb`](https://github.com/euroargodev/BBP_RTQC/blob/main/check_tests.ipynb)
and
[`test_json.ipynb`](https://github.com/euroargodev/BBP_RTQC/blob/main/test_json.ipynb)
ere used respectively to assert the tests and plot the example files inputs and outputs.

 




# TO-DO LIST
- Add Animal-Spike test
- Add non-medfilt high BBP test? 
