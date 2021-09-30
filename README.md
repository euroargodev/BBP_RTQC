# Introduction
This repository holds a new set of real-time quality-control (RTQC) test for particulate optical backscattering (BBP) data collected by BGC-Argo floats.

# A bit of history
A draft set of BBP RTQC tests and their impact on the GDAC data were presented at the ADMT21 meeting in 2020: https://drive.google.com/file/d/1Kkof40Cj5zWuVHQiRpudBRybXW6hJ4VQ/view?usp=sharing

A group of ADMT people interested in BBP had a post-ADMT21 meeting to discuss the proposed test further. I've prepared another presentation with the feedback/notes/decisions taken: https://drive.google.com/file/d/1JxdmBCS2MhG6qzmHwpmTCb-D5D3uX3qC/view?usp=sharing

The current version (30-Sep-2021) of the tests has not been presented to the community yet.

# Code
The code is written in Python3 or as Jupyter Notebooks.

## Test functions
The function `BBP_RTQC.py` contains all the proposed tests, besides other functions that are needed to QC the data.

## Global variables
`BBP_RTQC_global_vars.py`
`BBP_RTQC_paths.py`

## How to run tests on GDAC profiles
`run_BBP_RTQC.ipynb`

## How to plot and synthesize results
`bgc_argo_RTQC_plot_results.ipynb`

## Asserting BBP RTQC tests
`BBP_RTQC_example_tests.json`
`check_tests.ipynb`
`test_json.ipynb`
`prep_json.ipynb`
 




# TO-DO LIST
- Rewrite README.md file to describe current status of repo 
- Add Animal-Spike test 
