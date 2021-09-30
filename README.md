# Introduction
This repository holds a new set of real-time quality-control (RTQC) test for particulate optical backscattering (BBP) data collected by BGC-Argo floats.

# A bit of history
A draft set of BBP RTQC tests and their impact on the GDAC data were presented at the ADMT21 meeting in 2020: https://drive.google.com/file/d/1Kkof40Cj5zWuVHQiRpudBRybXW6hJ4VQ/view?usp=sharing

A group of ADMT people interested in BBP had a post-ADMT21 meeting to discuss the proposed test further. I've prepared another presentation with the feedback/notes/decisions taken: https://drive.google.com/file/d/1JxdmBCS2MhG6qzmHwpmTCb-D5D3uX3qC/view?usp=sharing

The current version (30-Sep-2021) of the tests has not been presented to the community yet.

# Code
The code is written in Python3 or as Jupyter Notebooks.

## Test functions
`BBP_RTQC.py`: This sript contains all the proposed tests, besides other functions that are needed to QC the data. It is loaded as a module by various notebooks.

## Global variables
`BBP_RTQC_global_vars.py`: This script contains all global variables needed by the different notebooks. It is loaded by various notebooks.

`BBP_RTQC_paths.py`: This script contains the paths to the data, the workign directory, and the directories where the pickled results and the plots are stored. The paths currently stored here are mine, so you'll have to update these to the ones in your machine.

## How to run tests on GDAC profiles
`run_BBP_RTQC.ipynb`: This is the main notebook used to apply the BBP RTQC tests. After importing the main modules used, it asserts all tests, and creates a list of DAC/WMO to be processed. It then remove any existing pickled and plot files. Finally, it runs the tests on each float in parallel (using 7 core CPUs, i.e., `n_jobs`: you will need to update this argument based on your machine).
There are also a number of commented cells that can be used run the tests sequentially, while being able to read the output (set `VERBOSE=True`), and to create a directory where all plots from a given test can be stored. 

## How to plot and synthesize results
`bgc_argo_RTQC_plot_results.ipynb`: This notebook is used to present a summary of the results of each test on the GDAC data. After reading all the pickled files, it extract statistics on each RTQC test 

## Asserting BBP RTQC tests
`BBP_RTQC_example_tests.json`

`check_tests.ipynb`

`test_json.ipynb`

`prep_json.ipynb`
 




# TO-DO LIST
- Rewrite README.md file to describe current status of repo 
- Add Animal-Spike test 
