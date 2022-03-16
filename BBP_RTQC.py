import numpy as np
import xarray as xr
import os
import glob

# Plotting
import matplotlib.pyplot as plt
import ipdb

import warnings
import pickle
import gc

from BBP_RTQC_global_vars import *
from BBP_RTQC_paths import *

import json
import re

warnings.filterwarnings('ignore')

def ini_flags(BBP700):
    # initialise output array of flags for prep_json and test_tests
    BBP700_QC_1st_failed_test = dict.fromkeys(tests.keys())
    for ikey in BBP700_QC_1st_failed_test.keys():
        BBP700_QC_1st_failed_test[ikey] = np.full(shape=BBP700.shape, fill_value='0')
    return BBP700_QC_1st_failed_test


def test_tests(ia):
    # compare current results of tests against existing BBP_RTQC_example_tests.json
    f = open("BBP_RTQC_example_tests.json")  # open the text file
    t = json.load(f)  # interpret the text file and make it usable by python
    f.close()
    a = []
    for it, tmp in enumerate(t):
        a.append(json.loads(t[it]))

    # find index of code
    code = a[ia]['code']
    print(str(ia), " ", code)

    # create BBP700 array
    BBP = np.asarray(a[ia]['input']['BBP'])
    if code != 'H':
        BBPmf1 = np.asarray(a[ia]['input']['BBPmf1'])
    PRES = np.asarray(a[ia]['input']['PRES'])
    if code == 'G':
        maxPRES = np.asarray(a[ia]['input']['maxPRES'])
        PARK_PRES = np.asarray(a[ia]['input']['PARK_PRES'])
    if code == 'E':
        maxPRES = np.asarray(a[ia]['input']['maxPRES'])
    if code == 'H':
        COUNTS = np.asarray(a[ia]['input']['COUNTS'])

    # initialise BBP700_QC_1st_failed_test
    BBP_QC_failed_test = ini_flags(BBP)

    def myfunc(msg='assert OK'):
        print(msg)
        return True

    if code == 'A':
        #### Global range
        QC_FLAGS_OUT, BBP_QC_failed_test = BBP_Global_range_test(BBP, BBPmf1, PRES,
                                                                        np.ones(BBP.shape),
                                                                        BBP_QC_failed_test,
                                                                        'test_tests')
    elif code == 'A2':
        #### Global range
        QC_FLAGS_OUT, BBP_QC_failed_test = BBP_Global_range_test(BBP, BBPmf1, PRES,
                                                                        np.ones(BBP.shape),
                                                                        BBP_QC_failed_test,
                                                                        'test_tests')
    elif code == 'B':
        #### Noisy Profile
        QC_FLAGS_OUT, BBP_QC_failed_test, tmp = BBP_Noisy_profile_test(BBP, BBPmf1, PRES,
                                                                        np.ones(BBP.shape),
                                                                        BBP_QC_failed_test,
                                                                        'test_tests')
    elif code == 'C':
        #### High-Deep Value
        QC_FLAGS_OUT, BBP_QC_failed_test = BBP_High_Deep_Values_test(BBPmf1, PRES,
                                                                         np.ones(BBP.shape),
                                                                         BBP_QC_failed_test,
                                                                         'test_tests')
    # elif code == 'D':
    #     #### Surface Hook
    #     QC_FLAGS_OUT, BBP_QC_failed_test = BBP_Surface_hook_test(BBP, BBPmf1, PRES,
    #                                                                      np.ones(BBP.shape),
    #                                                                      BBP_QC_failed_test,
    #                                                                      'test_tests')
    elif code == 'E':
        #### Missing Data
        QC_FLAGS_OUT, BBP_QC_failed_test = BBP_Missing_Data_test(BBP, PRES, maxPRES,
                                                                         np.ones(BBP.shape),
                                                                         BBP_QC_failed_test,
                                                                         'test_tests')
    # elif code == 'F':
    #     #### Negative non-surface
    #     QC_FLAGS_OUT, BBP_QC_failed_test = BBP_Negative_nonsurface_test(BBP, PRES,
    #                                                                             np.ones(BBP.shape),
    #                                                                             BBP_QC_failed_test,
    #                                                                             'test_tests')
    elif code == 'G':
        #### Parking Hook
        QC_FLAGS_OUT, BBP_QC_failed_test = BBP_Parking_hook_test(BBP, BBPmf1, PRES, maxPRES, PARK_PRES,
                                                                        np.ones(BBP.shape),
                                                                        BBP_QC_failed_test,
                                                                        'test_tests')
    # elif code == 'H':
    #     #### Stuck value
    #     QC_FLAGS_OUT, BBP_QC_failed_test = BBP_Stuck_Value_test(COUNTS, BBP, PRES,
    #                                                                 np.ones(BBP.shape),
    #                                                                 BBP_QC_failed_test,
    #                                                                 'test_tests')


    assert np.all(QC_FLAGS_OUT == a[ia]['output']['flags_out']) and myfunc(tests[code] + ' / ' + a[ia]['specifics'] + ': test succeded.'), 'Assertion error for ' + tests[code]


def test_BBP_RTQC():
    '''
    function to run example tests
    '''

    # read json file
    f = open("BBP_RTQC_example_tests.json")  # open the text file
    t = json.load(f)  # interpret the text file and make it usable by python
    f.close()

    # store json data in list
    a = []
    for it, tmp in enumerate(t):
        a.append(json.loads(t[it]))

    # assert tests
    for ia in range(len(a)):
        test_tests(ia)

    return


# medfilt1 function similar to Octave's that does not bias extremes of dataset towards zero
def medfilt1(data, kernel_size, endcorrection='shrinkkernel'):
     """One-dimensional median filter"""
     halfkernel = int(kernel_size/2)
     data = np.asarray(data)

     filtered_data = np.empty(data.shape)
     filtered_data[:] = np.nan

     for n in range(len(data)):
         i1 = np.nanmax([0, n-halfkernel])
         i2 = np.nanmin([len(data), n+halfkernel+1])
         filtered_data[n] = np.nanmedian(data[i1:i2])

     return filtered_data



# apply QC flag to array with all flags
def apply_qc(QC_Flags, ISBAD, QC, QC_1st_failed_test, QC_TEST_CODE):

    # find which part of the QC_Flag[ISBAD] array needs to be updated with the new flag
    i2flag = np.where( QC_Flags[ISBAD] < QC )[0]  # find where the existing flag is lower than the new flag (cannot lower existing flags)

    # apply flag
    QC_Flags[ISBAD[i2flag]] = QC

    # record which test changed the flag
    QC_1st_failed_test[QC_TEST_CODE][ISBAD] = QC_TEST_CODE

    return QC_Flags, QC_1st_failed_test


# function to define adaptive median filtering based on Christina Schallemberg's suggestion for CHLA
def adaptive_medfilt1(x, y, PLOT=False):
    # applies a median filtering followin Christina Schlallemberb's recommendations
    # x is PRES
    # y is BBP
    
    
#     x = PRES[innan]
#     y = BBP700[innan]

    # compute x resolution
    xres = np.diff(x)

    # initialise medfiltered array
    ymf = np.zeros(y.shape)*np.nan

    ir_LT1 = np.where(xres<1)[0]
    if np.any(ir_LT1):
        win_LT1 = 11.
        ymf[ir_LT1] = medfilt1(y[ir_LT1], win_LT1)   

    ir_13 = np.where((xres>=1) & (xres<=3))[0]
    if np.any(ir_13):
        win_13 = 7.
        ymf[ir_13] = medfilt1(y[ir_13], win_13)   

    ir_GT3 = np.where(xres>3)[0]
    if np.any(ir_GT3):
        win_GT3 = 5.
        ymf[ir_GT3] = medfilt1(y[ir_GT3], win_GT3)   

    if PLOT:
        plt.plot(np.log10(y), x, 'o')
        plt.plot(np.log10(ymf), x, 'r-')
    
    return ymf


# # Implement different RTQC for BBP data using B-files

# ## Tests that do not remove the entire profile

##################################################################
##################################################################
# def BBP_Refined_range_test(BBP, BBPmf1, PRES, QC_Flags, QC_1st_failed_test,
#                           fn, PLOT=False, SAVEPLOT=False, VERBOSE=False):
#     # BBP: nparray with all BBP data
#     # BBPmf1: median-filtered BBP data
#     # QC_Flags: array with QC flags
#     # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
#     # fn: name of corresponding B-file
#     # PLOT: flag to plot results
#     # SAVEPLOT: flag to save plot
#     # VERBOSE: flag to display verbose output
#
#     # Objective: To flag data points or profiles outside an expected range of BBP values.
#     # The expected range is defined by two extrema: A_MIN_BBP700 = 0 m-1 and A_MAX_BBP700 = 0.01 m-1.
#     # A_MIN_BBP700 is defined to flag negative values, while A_MAX_BBP700 is a conservative
#     # estimate of the maximum BBP to be expected in the open ocean, based on statistics of
#     # satellite and BGC-Argo data (Bisson et al., 2019).
#     #
#     # Implementation: The test is implemented on data that have been median filtered (to remove spikes).
#     #
#     # Flagging: A QC flag of 3 is assigned to data points that fall above A_MAX_BBP700, while the
#     # entire profile is flagged with QC = 3 if any data point falls below A_MIN_BBP700
#     # (this is to reflect the more serious condition of having negative median filtered data in a profile).
#     # __________________________________________________________________________________________
#     #
#
#     FAILED = False
#
#     QC = 3
#     QC_TEST_CODE = 'A' # or 'A2' if negative medfilt1 value is found
#     ISBAD = np.array([])  # index of where flags should be applied in the profile
#
#     # this is the test
#     ISBAD = np.where( (BBPmf1 > A_MAX_BBP700) | (BBPmf1 < A_MIN_BBP700) )[0]
#
#     if ISBAD.size != 0:# If ISBAD is not empty
#         FAILED = True
#         # flag entire profile if any negative value is found
#         if np.any(BBPmf1 < A_MIN_BBP700):
#             if VERBOSE:
#                 print('negative median-filtered BBP: flagging all profile')
#             QC_TEST_CODE = 'A2'
#             ISBAD = np.where(BBPmf1)[0]
#         # apply flag
#         QC_Flags, QC_1st_failed_test = apply_qc(QC_Flags, ISBAD, QC, QC_1st_failed_test, QC_TEST_CODE)
#
#         if VERBOSE:
#             print('Failed Global_Range_test')
#             print('applying QC=' + str(QC) + '...')
#
#     if (PLOT) & (FAILED):
#         plot_failed_QC_test(BBP, BBPmf1, PRES, ISBAD, QC_Flags, QC_1st_failed_test[QC_TEST_CODE], QC_TEST_CODE,
#                             fn, SAVEPLOT, VERBOSE)
#
#     return QC_Flags, QC_1st_failed_test

def BBP_Negative_BBP_test(BBP, PRES, QC_Flags, QC_1st_failed_test,
                          fn, PLOT=False, SAVEPLOT=False, VERBOSE=False):
    # BBP: nparray with all BBP data
    # BBPmf1: median-filtered BBP data
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    # PLOT: flag to plot results
    # SAVEPLOT: flag to save plot
    # VERBOSE: flag to display verbose output

    # Objective: To flag data points or profiles outside an expected range of BBP values.
    # The expected range is defined by two extrema: A_MIN_BBP700 = 0 m-1 and A_MAX_BBP700 = 0.01 m-1.
    # A_MIN_BBP700 is defined to flag negative values, while A_MAX_BBP700 is a conservative
    # estimate of the maximum BBP to be expected in the open ocean, based on statistics of
    # satellite and BGC-Argo data (Bisson et al., 2019).
    #
    # Implementation: The test is implemented on data that have been median filtered (to remove spikes).
    #
    # Flagging: A QC flag of 3 is assigned to data points that fall above A_MAX_BBP700, while the
    # entire profile is flagged with QC = 3 if any data point falls below A_MIN_BBP700
    # (this is to reflect the more serious condition of having negative median filtered data in a profile).
    # __________________________________________________________________________________________
    #

    FAILED = False

    QC_TEST_CODE = 'A' # or 'A2' if negative medfilt1 value is found

    # this is the test
    iLT5dbar = np.where( PRES < 5 )[0] # index for data shallower than 5 dbar
    ISBAD = np.where( BBP < A_MIN_BBP700 )[0] # first fill in all ISBAD indices where BBP < threshold
    ISBAD_gt5dbar = [x for x in ISBAD if x not in iLT5dbar] #  select only ISBAD indices deeper than 5 dbar
    ISBAD_lt5dbar = [x for x in ISBAD if x in iLT5dbar] # select only ISBAD indices shallower than 5 dbar

    if len(ISBAD_gt5dbar) != 0:# If ISBAD_gt5dbar is not empty
        FAILED = True
        QC = 4
        QC_TEST_CODE = 'A2'
        ISBAD = np.where(BBP)[0] # flag entire profile

        # apply flag
        QC_Flags, QC_1st_failed_test = apply_qc(QC_Flags, ISBAD, QC, QC_1st_failed_test, QC_TEST_CODE)

        if VERBOSE:
            print('Failed Global_Range_test')
            print('applying QC=' + str(QC) + '...')


    if len(ISBAD_lt5dbar) > len(ISBAD_gt5dbar): # if there are bad points only at PRES <5 dbar
        FAILED = True
        QC = 4
        QC_TEST_CODE = 'A'

        ISBAD = np.asarray(ISBAD_lt5dbar) # flag only negative values shallower than 5 dbar

        # apply flag
        QC_Flags, QC_1st_failed_test = apply_qc(QC_Flags, ISBAD, QC, QC_1st_failed_test, QC_TEST_CODE)

        if VERBOSE:
            print('Failed Global_Range_test')
            print('applying QC=' + str(QC) + '...')


    if (PLOT) & (FAILED):
        plot_failed_QC_test(BBP, BBP, PRES, ISBAD, QC_Flags, QC_1st_failed_test[QC_TEST_CODE], QC_TEST_CODE,
                            fn, SAVEPLOT, VERBOSE)

    return QC_Flags, QC_1st_failed_test

##################################################################
##################################################################
def BBP_Noisy_profile_test(BBP, BBPmf1, PRES, QC_Flags, QC_1st_failed_test,
                           fn, PLOT=False, SAVEPLOT=False, VERBOSE=False):
    # BBP: nparray with all BBP data
    # BBPmf1: smooth BBP array (medfilt1(BBP700, 31)
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    # PLOT: flag to plot results
    # SAVEPLOT: flag to save plot
    # VERBOSE: flag to display verbose output

    # Objective: To flag profiles that are affected by noisy data. This noise could 
    # indicate sensor malfunctioning, some animal spikes, or other anomalous conditions.
    #
    # Implementation: The absolute residuals between the median filtered BBP and the 
    # raw BBP values are computed below a pressure threshold B_PRES_THRESH = 100 dbar 
    # (this is to avoid surface data, where spikes are more common and generate false 
    # positives). The test fails if residuals with values above B_RES_THRESHOLD = 0.0005 m-1 
    # occur in at least B_FRACTION_OF_PROFILE_THAT_IS_OUTLIER = 10% of the profile. 
    # These threshold values were selected after visual inspection of flagged profiles.
    #
    # Flagging: If the test fails, a QC flag of 3 is assigned to the entire profile.
    #
    # __________________________________________________________________________________________

    FAILED = False

    QC = 3 # flag to apply if the result of the test is true
    QC_TEST_CODE = 'B'
    ISBAD = np.array([]) # flag for noisy profile

    res = np.empty(BBP.shape)
    res[:] = np.nan

    innan = np.where(~np.isnan(BBP))[0]

    if len(innan)>10: # if we have at least 10 points in the profile

        #new constraint: noise should be below 100 dbars
        iPRES = np.where(PRES[innan] > B_PRES_THRESH)[0]

        res[innan] = np.abs(BBP[innan]-BBPmf1[innan])
        ioutliers = np.where(abs(res[innan][iPRES]) > B_RES_THRESHOLD)[0] # index of where the rel res are greater than the threshold

        if len(ioutliers)/len(innan) >= B_FRACTION_OF_PROFILE_THAT_IS_OUTLIER: # this is the actual test: is there more than a certain fraction of points that are noisy?
            ISBAD = ioutliers

    # update QC_Flags to 3 when bad profiles are found
    if ISBAD.size != 0:
        FAILED = True
        # apply flag
        QC_Flags, QC_1st_failed_test = apply_qc(QC_Flags, np.where(BBP)[0], QC, QC_1st_failed_test, QC_TEST_CODE)# np.where(BBP)[0] is used to flag the entire profile

        if VERBOSE:
            print('Failed BBP_Noisy_Profile_test')
            print('applying QC=' + str(QC) + '...')

    if (PLOT) & (FAILED):
        plot_failed_QC_test(res, res*0., PRES, ISBAD, QC_Flags, QC_1st_failed_test[QC_TEST_CODE], QC_TEST_CODE,
                            fn, SAVEPLOT, VERBOSE)


    return QC_Flags, QC_1st_failed_test, res

##################################################################
##################################################################
def BBP_High_Deep_Values_test(BBPmf1, PRES, QC_Flags, QC_1st_failed_test,
                              fn, PLOT=False, SAVEPLOT=False, VERBOSE=False):
    # BBP: nparray with all BBP data
    # BBPmf1: smooth BBP array (medfilt1(BBP700, 31)
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    # PLOT: flag to plot results
    # SAVEPLOT: flag to save plot
    # VERBOSE: flag to display verbose output

    # Objective: To flag profiles with anomalously high BBP values at depth. 
    # These high values at depth could indicate a variety of problems, including 
    # biofouling, incorrect calibration coefficients, sensor malfunctioning, etc. 
    # A threshold value of 0.0005 m-1 was selected that is half of the value typical
    # for surface BBP in the oligotrophic ocean: median-filtered BBP data at depth
    # are expected to be considerably lower than this threshold value.
    #
    # Implementation: This tests fails if the median-filtered BBP profile has at 
    # least a certain number (C_N_of_ANOM_POINTS = 5) of anomalous points 
    # (medfilt(BBP700) > C_DEEP_BBP700_THRESH = 0.0005 m-1) below a threshold 
    # depth (C_DEPTH_THRESH = 700 dbar). Note that this test can only be implemented 
    # if the profile reaches a maximum pressure greater than 700 dbar. Variables 
    # in capital letters represent parameters used by each test that can be modified, if needed.
    #
    # Flagging: If the test fails, a QC flag of 3 is applied to the entire profile. 
    # High deep BBP values can result from a variety of reasons, including natural causes, 
    # in which case data might be set to good quality during DMQC. Therefore, we decided
    # to use QC=3 and to revise these profiles during DMQC.
    #
    # __________________________________________________________________________________________

    FAILED = False

    QC = 3; # flag to apply if the result of the test is true
    QC_TEST_CODE = 'C'
    ISBAD = np.array([]) # flag for noisy profile

    # this is the test
    iDEEP = np.where(PRES > C_DEPTH_THRESH)[0]
    if (np.nanmedian(BBPmf1[iDEEP]) > C_DEEP_BBP700_THRESH) & (len(BBPmf1[iDEEP]) >= C_N_of_ANOM_POINTS):
        ISBAD = np.where(BBPmf1)[0]

    if ISBAD.size != 0: # if ISBAD, then apply QC_flag=3
        FAILED = True
        # apply flag
        QC_Flags, QC_1st_failed_test = apply_qc(QC_Flags, ISBAD, QC, QC_1st_failed_test, QC_TEST_CODE)# np.where(BBP)[0] is used to flag the entire profile

        if VERBOSE:
            print('Failed High_Deep_Values_test')
            print('applying QC=' + str(QC) + '...')

    if (PLOT) & (FAILED):
                plot_failed_QC_test(BBPmf1, BBPmf1, PRES, ISBAD, QC_Flags, QC_1st_failed_test[QC_TEST_CODE],
                                    QC_TEST_CODE, fn, SAVEPLOT, VERBOSE)



    return QC_Flags, QC_1st_failed_test

##################################################################
##################################################################
def BBP_Missing_Data_test(BBP, PRES, maxPRES, QC_Flags, QC_1st_failed_test,
                          fn, PLOT=False, SAVEPLOT=False, VERBOSE=False):
    # BBP: nparray with all BBP data
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    # PLOT: flag to plot results
    # SAVEPLOT: flag to save plot
    # VERBOSE: flag to display verbose output
    
    # Objective: To detect and flag profiles that have a large fraction of 
    # missing data. Missing data could indicate shallow or incomplete profiles.
    #
    # Implementation: The upper 1000 dbar of the profile are divided into 10 
    # pressure bins with the following lower boundaries (all in dbar): 
    # 50, 156, 261, 367, 472, 578,  683, 789, 894, 1000. For example, 
    # the first bin covers the pressure range [0, 50), the second [51, 156), 
    # etc. The test fails if any of the bins contains fewer data points than MIN_N_PERBIN = 1.
    # 
    # Flagging: Different flags are assigned depending on how many bins are empty.
    # If only one bin contains data or the profile has no data at all, a QC flag of 4 
    # is applied to the entire profile. This condition may indicate a malfunctioning 
    # sensor or a profile that is so shallow that it is too difficult to quality control in real time.
    # If there are bins with missing data, but the number of bins with data is 
    # greater than one, then a QC flag of 3 is assigned to the entire profile.
    # __________________________________________________________________________________________

    FAILED = False

    QC_all = [np.nan, np.nan, np.nan]
    QC_all[0] = 3 # 2 flag to apply if shallow profile
    QC_all[1] = 4 # flag to apply if the result of the test is true only in one bin
    QC_all[2] = 3 # flag to apply if the result of the test is true elsewhere

    QC_TEST_CODE = 'E'
    ISBAD = np.array([])  # index of where flags should be applied in the profile


    # bin the profile into 100-dbars bins
    bins = np.linspace(50, 1000, 10) # create 10 bins between 0 and 1000 dbars
    bin_counts = np.zeros(bins.shape) # initialise array with number of counts in each bin
    for i in range(len(bins)):
        if i == 0:
            bin_counts[i] = len(np.where(PRES < bins[i])[0])
        else:
            bin_counts[i] = len(np.where((PRES >= bins[i-1]) & (PRES < bins[i]))[0])

    # check if there are bins with missing data
    if np.any(np.nonzero(bin_counts < E_MIN_N_PERBIN)[0]):
        isbad4plot = np.where(bin_counts < E_MIN_N_PERBIN)[0]
        ISBAD = np.where(BBP)[0] # flag the entire profile

        # find which bins contain data
        nonempty = np.where(bin_counts > 0)[0] # index of bins that contain data points

        # select which flag to use
        if nonempty.size != 0:

            # if shallow profile
            if (maxPRES < E_MAXPRES) & (len(np.nonzero(bin_counts > E_MIN_N_PERBIN)[0]) > 1):
                if VERBOSE: print("shallow profile: QC=" + str(QC_all[0])) # with test
                QC = QC_all[0]

            # if there is only one bin with data then
            elif len(np.nonzero(bin_counts > E_MIN_N_PERBIN)[0]) == 1:  # with test
                if VERBOSE: print("data only in one bin: QC=" + str(QC_all[1]))
                QC = QC_all[1]

            # if missing data profile
            else:
                if VERBOSE: print("missing data: QC=" + str(QC_all[2])) # with test
                QC = QC_all[2]

        else: # this is for when we have no data at all, then
            if VERBOSE: print("no data at all: QC=" + str(QC_all[1])) # with test
            QC = QC_all[1]

    if ISBAD.size != 0: # if ISBAD, then apply QC_flag
        FAILED = True
        # apply flag
        QC_Flags, QC_1st_failed_test = apply_qc(QC_Flags, ISBAD, QC, QC_1st_failed_test, QC_TEST_CODE)# np.where(BBP)[0] is used to flag the entire profile

        if VERBOSE:
            print('Failed Missing_Data_test')
            print('applying QC=' + str(QC) + '...')

    if (PLOT) & (FAILED):
                plot_failed_QC_test(BBP, bin_counts, PRES, isbad4plot, QC_Flags, QC_1st_failed_test[QC_TEST_CODE],
                                    QC_TEST_CODE, fn, SAVEPLOT, VERBOSE)

    return QC_Flags, QC_1st_failed_test

##################################################################
##################################################################
def BBP_Parking_hook_test(BBP, BBPmf1, PRES, maxPRES, PARK_PRES, QC_Flags, QC_1st_failed_test,
                          fn, PLOT=False, SAVEPLOT=False, VERBOSE=False):
    # BBP: nparray with all BBP data
    # BBPmf1: nparray with medfilt BBP data
    # maxPRES: maximum pressure recorded in this profile
    # PARK_PRES: programmed parking pressure for this profile
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    # PLOT: flag to plot results
    # SAVEPLOT: flag to save plot
    # VERBOSE: flag to display verbose output
    
    # Objective: To flag data points near the parking pressure with 
    # anomalously high values, when the parking pressure is close to the 
    # maximum pressure of the profile. This could indicate that particles 
    # have accumulated on the sensor or the float and that are released when the float starts ascending.
    # 
    # Implementation: First the parking pressure (PARK_PRES) is extracted 
    # from the metadata file. Then, we verify that the vertical resolution of
    # the data near PARK_PRES is greater than G_DELTAPRES2 = 20 dbar: if it is not, 
    # the test cannot be applied to this profile. If the vertical resolution is sufficient, 
    # we verify that the maximum pressure of the profile is less than G_DELTAPRES0 (100 dbar) different 
    # from PARK_PRES (i.e., that PARK_PRES ~= max(PRES)), i.e., that the profile starts 
    # from the parking pressure. If it does, a pressure range iPRESmed 
    # (max(PRES) - G_DELTAPRES2 > PRES >= max(PRES) - G_DELTAPRES1, with G_DELTAPRES1 = 50 dbar) 
    # is defined over which the baseline for the test will be calculated. This baseline 
    # is computed as the median + G_DEV (with G_DEV =  0.0002 m-1). The test is implemented 
    # in the pressure range iPREStest (where PRES>= maxPRES - G_DELTAPRES1). The test fails 
    # if BBP within iPREStest is greater than the baseline.
    # 
    # Flagging: A QC flag of 4 is applied to the points that fail the test.
    # 
    # __________________________________________________________________________________________

    FAILED = False

    QC = 4
    QC_TEST_CODE = 'G'
    ISBAD = np.array([]) # flag for noisy profile

    if (np.isnan(maxPRES)) | (np.isnan(PARK_PRES)):
        if VERBOSE: print('WARNING:\nmaxPRES='+str(maxPRES)+' dbars,\nPARK_PRES='+str(PARK_PRES))
        ipdb.set_trace()


    # check that there are enough data to run the test
    imaxPRES = np.where(PRES == maxPRES)[0][0]
    deltaPRES = PRES[imaxPRES] - PRES[imaxPRES-1]
    if deltaPRES > G_DELTAPRES2:
        if VERBOSE: print('vertical resolution is too low to check for Parking Hook')
        return QC_Flags, QC_1st_failed_test


    # check if max PRES is 'close' (i.e., within 100 dbars) to PARK_PRES
    if abs(maxPRES - PARK_PRES) >= G_DELTAPRES0:
        return QC_Flags, QC_1st_failed_test


    # define PRES range over which to compute the baseline for the test
    iPRESmed = np.where((PRES >= maxPRES - G_DELTAPRES1 ) & (PRES < maxPRES - G_DELTAPRES2) )[0]
    # define PRES range over which to apply the test
    iPREStest = np.where((PRES >= maxPRES - G_DELTAPRES1 ))[0]

    # compute parameters to define baseline above which test fails
    medBBP = np.nanmedian(BBP[iPRESmed])
    baseline = medBBP + G_DEV

    # this is the test
    ibad = np.where(BBP[iPREStest] > baseline)[0]
    ISBAD = iPREStest[ibad]

    if ISBAD.size != 0: # If ISBAD is not empty
        FAILED = True
        # apply flag
        QC_Flags, QC_1st_failed_test = apply_qc(QC_Flags, ISBAD, QC, QC_1st_failed_test, QC_TEST_CODE)

        if VERBOSE:
            print('Failed Parking_hook_test')
            print('applying QC=' + str(QC) + '...')

    if (PLOT) & (FAILED):
        plot_failed_QC_test(BBP, BBP, PRES, ISBAD, QC_Flags, QC_1st_failed_test[QC_TEST_CODE], QC_TEST_CODE,
                            fn, SAVEPLOT, VERBOSE)

    return QC_Flags, QC_1st_failed_test

##################################################################
##################################################################
# function to plot results of applying test to dataset
def plot_failed_QC_test(BBP, BBPmf1, PRES, ISBAD, QC_Flags, QC_1st_failed_test, QC_TEST_CODE,
                        fn, SAVEPLOT=False, VERBOSE=False):
    # BBP: nparray with all BBP data
    # BBPmf1: median-filtered BBP data
    # ISBAD: index marking the data that failed the QC test
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # QC_TEST_CODE: code of the failed test
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1,1,1)

    innan = np.nonzero(~np.isnan(BBP))

    # check that there are enough data to plot
    if len(BBP) < 2:
        if VERBOSE:
            print("not enough data to plot... exiting")
        return
    

    if (QC_TEST_CODE != "0") & (QC_TEST_CODE != "E"):
        ax1.plot(BBPmf1[ISBAD], PRES[ISBAD], 'o', ms=10, color='r', mfc='r', alpha=0.7, zorder=60)
        ax1.plot(BBP[innan], PRES[innan], 'o-', ms=3, color='k', mfc='none', alpha=0.7) # <<<<<<<<<<<<<<<<<<

    if QC_TEST_CODE != 'E':
        ax1.plot(BBP[innan], PRES[innan], 'o-', ms=3, color='k', mfc='none', alpha=0.7) # <<<<<<<<<<<<<<<<<<
        ax1.plot(BBPmf1[innan], PRES[innan], '-', color='#41F11D', mfc='none', alpha=0.7)
    

    # test-specific additions
    if (QC_TEST_CODE == "A") | (QC_TEST_CODE == "0") | (QC_TEST_CODE == "D") | (QC_TEST_CODE == "F") | (QC_TEST_CODE == "G"):
        ax1.plot(BBP[innan], PRES[innan], 'o-', ms=3, color='k', mfc='none', alpha=0.7) # <<<<<<<<<<<<<<<<<<
        ax1.plot(BBPmf1[innan], PRES[innan], '-', color='#41F11D', mfc='none', alpha=0.7)
        ax1.plot(A_MAX_BBP700*np.ones(2), [-5, 2000], '--', color='r', mfc='none', alpha=0.7)
        ax1.plot(A_MIN_BBP700*np.ones(2), [-5, 2000], '--', color='r', mfc='none', alpha=0.7)
        
    if QC_TEST_CODE == "C":
        ax1.plot(C_DEEP_BBP700_THRESH*np.ones(2), [-5, 2000], '--', color='r', mfc='none', alpha=0.7)

    if QC_TEST_CODE != 'B':
        ax1.set_xlim((-0.001, 0.015))
    
    if QC_TEST_CODE == "A":
        ax1.set_xlim((-0.001, 0.045))

    elif QC_TEST_CODE =='B':
        ax1.plot(B_RES_THRESHOLD*np.ones(2), [-5, 2000], '--', color='r', mfc='none', alpha=0.7)
        ax1.set_xlim( (1e-6, 1e-2 ) )
        ax1.set_xscale('log')

    if QC_TEST_CODE == "E":
        bin_counts = BBPmf1+10.
        ax1.cla()
        bins = np.linspace(50, 1000, 10)
        ax1.barh(bins-50, bin_counts/2000., height=97, zorder=0 )
        ifail = np.where(bin_counts<10+1)
        ax1.barh(bins[ifail]-50, bin_counts[ifail]/2000., height=97, color='r', zorder=1 )
        ax1.plot(BBP, PRES, 'k.-', zorder=60)   
        

    ax1.grid('on')

    ax1.set_ylim([-5, 2000])    
    ax1.invert_yaxis()

    ax1.set_xlabel('BBP [1/m]', fontsize=20)
    ax1.set_ylabel('PRES [dbars]', fontsize=20)
    ax1.set_title('QC='+QC_TEST_CODE+" "+fn.split('/')[-1], fontsize=20, color='r', fontweight='bold')

    if SAVEPLOT:
        if VERBOSE:
            print("saving plot...")
        fname = DIR_PLOTS + "/" + fn.split('/')[-3] + "/" + fn.split('/')[-4] + "_" + fn.split('/')[-1] + "_" + QC_TEST_CODE+ ".png"
        fig.savefig(fname, dpi = 75) 

    # minimise memory leaks
    plt.close(fig)
    gc.collect()
    
    return


# function to extract basic data from Argo NetCDF meta file
def rd_WMOmeta(iwmo, VERBOSE):
    # read meta file to extract info on PARKING DEPTH
    fn = glob.glob(MAIN_DIR + 'dac/' + iwmo + '/*meta.nc')[0]
    ds_config = xr.open_dataset(fn)


    ## extract info on SENSOR
    if not np.any(ds_config.SENSOR.astype('str').str.contains('BACKSCATTERINGMETER_BBP700')):
        if VERBOSE:
            print("----this float does not have SENSOR metadata")
        SENSOR_MODEL = 'no metadata'
        SENSOR_MAKER = 'no metadata'
        SENSOR_SERIAL_NO = 'no metadata'
        SCALE_BACKSCATTERING700 = 126
        DARK_BACKSCATTERING700 = 126
        KHI_BACKSCATTERING700 = 126
    else:    
        iBBPsensor = np.where(ds_config.SENSOR.astype('str').str.contains('BACKSCATTERINGMETER_BBP700'))[0][0] # find index of BBP meter
        SENSOR_MODEL = ds_config.SENSOR_MODEL[iBBPsensor].astype('str').values
        SENSOR_MAKER = ds_config.SENSOR_MAKER[iBBPsensor].astype('str').values
        SENSOR_SERIAL_NO = ds_config.SENSOR_SERIAL_NO[iBBPsensor].astype('str').values

        # read PRE-DEPLOYMENT calibration coefficients
        if re.search("BACKSCATTERING700", str(ds_config.PREDEPLOYMENT_CALIB_COEFFICIENT.astype('str').values)): # check that CAL COEFFS are stored
            iBBP700cal = np.where(ds_config.PREDEPLOYMENT_CALIB_COEFFICIENT.astype('str').str.contains('BACKSCATTERING700'))[0][0]
            calcoeff_string = ds_config.PREDEPLOYMENT_CALIB_COEFFICIENT[iBBP700cal].astype('str').values
            calcoeff_string = np.char.strip(calcoeff_string).item()

            # check what delimiter was used
            if re.search(";", calcoeff_string):
                delim = ";"
            elif re.search(",", calcoeff_string):
                delim = ","
            else:
                print("delimiter not found")

            for text in calcoeff_string.split(delim):
                if "DARK_BACKSCATTERING700" in text:
                    DARK_BACKSCATTERING700 = float(text.split("=")[-1])
                elif "khi" in text:
                    KHI_BACKSCATTERING700 = float(text.split("=")[-1])
                elif "SCALE_BACKSCATTERING700" in text:
                    SCALE_BACKSCATTERING700 = float(text.split("=")[-1])

        else:
            if VERBOSE:
                print("no calibration coefficients found")
            SCALE_BACKSCATTERING700 = 125 # different flag from above to differentiate them
            DARK_BACKSCATTERING700 = 125
            KHI_BACKSCATTERING700 = 125

    if ds_config.PLATFORM_TYPE.values:
        PLATFORM_TYPE = str(ds_config.PLATFORM_TYPE.values.astype(str))
    else:
        PLATFORM_TYPE = 'no metadata'
        if VERBOSE:
            print("PLATFORM_TYPE=" + PLATFORM_TYPE)
    
    # extract CONFIG_MISSION_NUMBER from meta file
    miss_no_float = ds_config.CONFIG_MISSION_NUMBER.values
    
    return ds_config, SENSOR_MODEL, SENSOR_MAKER, SENSOR_SERIAL_NO, PLATFORM_TYPE, miss_no_float, \
           DARK_BACKSCATTERING700, SCALE_BACKSCATTERING700, KHI_BACKSCATTERING700


# read BBP and PRES
def rd_BBP(fn_p, miss_no_float, ds_config, VERBOSE=False):
    ds = xr.open_dataset(fn_p)

    # check if BBP700 is present
    v = set(ds.data_vars)
    if 'BBP700' not in v:
        if VERBOSE: print('no BBP700 for this cycle')
        ds.close()
        # set returned values to flag
        PRES = BBP700 = COUNTS = JULD = LAT = LON = BBP700mf1 = miss_no_prof = PARK_PRES = maxPRES = innan = -12345678
        return PRES, BBP700, JULD, LAT, LON, BBP700mf1, miss_no_prof, PARK_PRES, maxPRES, innan, COUNTS
        


    # find N_PROF where the BBP700 data are stored     
    tmp_bbp = ds.BBP700.values # note that BBP700[N_PROF,N_LEVELS]
    tmp = [np.any(~np.isnan(tmp_bbp[i][:])) for i in range(tmp_bbp.shape[0])] # find which of the different columns of tmp_bbp has at least one non-NaN element
    if np.any(tmp):
        N_PROF = np.where(tmp)[0][0]
    else:
        if VERBOSE: print("this profile has less than 5 data points: skipping ")
        # set returned values to flag
        PRES = BBP700 = COUNTS = JULD = LAT = LON = BBP700mf1 = miss_no_prof = PARK_PRES = maxPRES = innan = -12345678
        return PRES, BBP700, JULD, LAT, LON, BBP700mf1, miss_no_prof, PARK_PRES, maxPRES, innan, COUNTS

    COUNTS = ds.BETA_BACKSCATTERING700[N_PROF].values
    BBP700 = ds.BBP700[N_PROF].values
    
#    if 'PRES_ADJUSTED' in ds.keys():    
#        if VEBOSE: print('found PRES_ADJUSTED')
#        PRES = ds.PRES_ADJUSTED[N_PROF].values
#
#    else:
#        if VERBOSE: print('no PRES_ADJUSTED for this cycle')
#        ds.close()
#        # set returned values to flag
#        PRES = BBP700 = COUNTS = JULD = LAT = LON = BBP700mf1 = miss_no_prof = PARK_PRES = maxPRES = innan = -12345678
#        return PRES, BBP700, JULD, LAT, LON, BBP700mf1, miss_no_prof, PARK_PRES, maxPRES, innan, COUNTS

    PRES = ds.PRES[N_PROF].values
    JULD = ds.JULD[N_PROF].values
    LAT = ds.LATITUDE[N_PROF].values
    LON = ds.LONGITUDE[N_PROF].values

    innan = np.where(~np.isnan(BBP700))[0]

    # compute median filtered profile
    BBP700mf1 = np.zeros(BBP700.shape)*np.nan
    BBP700mf1[innan] = adaptive_medfilt1(PRES[innan], BBP700[innan])




######### needed for Parking-hook test #########################################  
    # Read Mission Number in profile to extract PARKING DEPTH
    miss_no_prof = ds.CONFIG_MISSION_NUMBER.values
#         if len(miss_no)>1:
#             if VERBOSE: print('WARNING: multiple ds.CONFIG_MISSION_NUMBER.values, \nchoosing the first')
#             miss_no = miss_no[0]

    if ~np.all(miss_no_prof==miss_no_prof[0]):
        print("different mission numbers in this profile")
        ipbd.set_trace()
    else:
        miss_no_prof = int(miss_no_prof[0])

    # find index of mission number in META file corresponding to profile
    if len(np.where(miss_no_float==miss_no_prof)[0])==0:
        if VERBOSE:
            print('the CONFIG_MISSION_NUMBER of the profile does not have a corresponding value in the META file')
        i_miss_no = 0            
    else:
        i_miss_no = np.where(miss_no_float==miss_no_prof)[0][0]


    # find Park Pressure in META file
    iParkPres = np.where(ds_config.CONFIG_PARAMETER_NAME.astype('str').str.contains('CONFIG_ParkPressure_dbar'))[0][0]
    PARK_PRES = ds_config.CONFIG_PARAMETER_VALUE.values[i_miss_no,iParkPres]
    if np.isnan(PARK_PRES):
        # assume PARK_PRES=1000 dbars
        PARK_PRES = 1000.
    maxPRES = np.nanmax(PRES)


    # close dataset
    ds.close()


    return PRES, BBP700, JULD, LAT, LON, BBP700mf1, miss_no_prof, PARK_PRES, maxPRES, innan, COUNTS


# function to apply tests and plot results (needed in function form for parallel processing)
def QC_wmo(iwmo, PLOT=False, SAVEPLOT=False, SAVEPKL=False, VERBOSE=False):
    #
    # # these are the tests and their codes
    # tests = {"A": "Global Range",
    #          "A2": "Global Range: negative",
    #          "B": "Noisy Profile",
    #          "C": "High-Deep Value",
    #          "D": "Surface Hook",
    #          "E": "Missing Data",
    #          "F": "Negative non-surface",
    #          "G": "Parking Hook",
    #          "H": "Stuck Value"
    #          }

    print(iwmo)

    if len(iwmo)==0:
        return

    # read meta file
    [ds_config, SENSOR_MODEL, SENSOR_MAKER, SENSOR_SERIAL_NO, PLATFORM_TYPE,
     miss_no_float, DARK_BACKSCATTERING700, SCALE_BACKSCATTERING700, KHI_BACKSCATTERING700] = rd_WMOmeta(iwmo, VERBOSE)
    
    # list single profiles
    fn2glob = MAIN_DIR + "dac/" + iwmo + "/profiles/" + "B*" + iwmo.split("/")[-1] + "*_[0-9][0-9][0-9].nc"
    fn_single_profiles = np.sort(glob.glob(fn2glob))

    if SAVEPLOT:
        if VERBOSE:
            print("Checking that dir is there, if not create it...")
        # create dir for output plots
        dout = DIR_PLOTS + "/" + fn_single_profiles[0].split('/')[-3]
        if not os.path.isdir(dout):
            os.mkdir(dout)
            if VERBOSE:
                print("created " + dout)
        else: # remove old plots + pkl file from this dir
            if VERBOSE:
                print("removing old plots in " + dout + "...")
                print(DIR_PLOTS + iwmo.split("/")[-1] + "/*.png")
            oldfn = glob.glob(DIR_PLOTS + iwmo.split("/")[-1] + "/*.png")
            [os.remove(i) for i in oldfn]   
            if VERBOSE:
                print("...done")

    if SAVEPKL: # remove existing pkl file from this dir
        if VERBOSE:
            print("removing old pickled files in " + dout + "...")
            print(DIR_PLOTS + iwmo.split("/")[-1] + "/*.pkl")
        oldfn = glob.glob(DIR_PLOTS + iwmo.split("/")[-1] + "/*.pkl")

        [os.remove(i) for i in oldfn]   
        if VERBOSE:
            print("...done")


    # initialise list that will store all data from this float
    all_PROFS = []

    for ifn_p, fn_p in enumerate(fn_single_profiles):
        if VERBOSE:
            print(fn_p)

        # read BBP and PRES + other data from profile file
        [ PRES, BBP700, JULD, LAT, LON, BBP700mf1, miss_no_prof, PARK_PRES, maxPRES, innan, COUNTS] = rd_BBP(fn_p, miss_no_float, ds_config, VERBOSE)
        if np.any(PRES==-12345678):
            continue

        # initialise arrays with QC flags[0,:] = 1 (good data)
        BBP700_QC_flags = np.zeros(BBP700.shape)+1
        BBP700_QC_1st_failed_test = dict.fromkeys(tests.keys())
        for ikey in BBP700_QC_1st_failed_test.keys():
            BBP700_QC_1st_failed_test[ikey] = np.full(shape=BBP700.shape, fill_value='0')

        # # Plot original profile even if no QC flag is raisef
#        if 'coriolis' in fn_p:
#            plot_failed_QC_test(BBP700, BBP700mf1, PRES, BBP700*np.nan, BBP700_QC_flags, BBP700_QC_1st_failed_test, '0', fn_p, SAVEPLOT, VERBOSE)
#

        # GLOBAL-RANGE TEST for BBP700
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Negative_BBP_test(BBP700, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, PLOT, SAVEPLOT, VERBOSE)

        # # SURFACE-HOOK TEST for BBP700
        # BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Surface_hook_test(BBP700, BBP700mf1, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, PLOT, SAVEPLOT, VERBOSE)

        # PARKING-HOOK TEST for BBP700
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Parking_hook_test(BBP700, BBP700mf1, PRES, maxPRES, PARK_PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, PLOT, SAVEPLOT, VERBOSE)

        # BBP_NOISY_PROFILE TEST
        BBP700_QC_flag, BBP700_QC_1st_failed_test, rel_res = BBP_Noisy_profile_test(BBP700, BBP700mf1, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, PLOT, SAVEPLOT, VERBOSE)
        if np.any(BBP700_QC_1st_failed_test["B"]!='0'):
            plot_failed_QC_test(BBP700, BBP700mf1, PRES, BBP700 * np.nan, BBP700_QC_flags, BBP700_QC_1st_failed_test,
                                '0', fn_p, SAVEPLOT, VERBOSE)

        # HIGH_DEEP_VALUES TEST
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_High_Deep_Values_test(BBP700mf1, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, PLOT, SAVEPLOT, VERBOSE)
        
        # MISSING_DATA TEST
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Missing_Data_test(BBP700, PRES, maxPRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, PLOT, SAVEPLOT, VERBOSE)
        
        # # NEGATIVE NON-SURFACE TEST
        # BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Negative_nonsurface_test(BBP700, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, PLOT, SAVEPLOT, VERBOSE)

        # # STUCK-VALUE TEST
        # BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Stuck_Value_test(COUNTS, BBP700, PRES, BBP700_QC_flags,
        #                                                                          BBP700_QC_1st_failed_test, fn_p, PLOT,
        #                                                                          SAVEPLOT, VERBOSE)

        # create dictionary with results
        prof = {}
        prof["BBP700_QC_1st_failed_test"] = dict.fromkeys(tests.keys())
        for ikey in prof["BBP700_QC_1st_failed_test"]:
            prof["BBP700_QC_1st_failed_test"][ikey] = BBP700_QC_1st_failed_test[ikey][innan]
        # add other keys in dictionary
        prof["JULD"] = JULD
        prof["LAT"] = LAT.tolist()
        prof["LON"] = LON.tolist()
        prof["PRES"] = PRES[innan]
        prof["BBP700"] = BBP700[innan]
        prof["BBP700_QC_flag"] = BBP700_QC_flag[innan]


        all_PROFS.extend([prof])
        del prof
        
    # add META variables at the end
    all_PROFS.extend([{'PARK_PRES':PARK_PRES, 
                       'SENSOR_MODEL':SENSOR_MODEL, 
                       'SENSOR_MAKER':SENSOR_MAKER, 
                       'SENSOR_SERIAL_NO':SENSOR_SERIAL_NO, 
                       'PLATFORM_TYPE':PLATFORM_TYPE,
                       'iWMO': iwmo,
                       'DARK_BACKSCATTERING700' : DARK_BACKSCATTERING700,
                       'SCALE_BACKSCATTERING700' : SCALE_BACKSCATTERING700,
                       'KHI_BACKSCATTERING700' : KHI_BACKSCATTERING700
                        }])

    if SAVEPKL:
        # save results in pickled file (https://www.datacamp.com/community/tutorials/pickle-python-tutorial)
        fname = DIR_PLOTS + fn_p.split('/')[-3] + "/" + fn_p.split('/')[-4] + "_" + fn_p.split('/')[-1].split('.')[0].split('_')[0] +  ".pkl"
        fnout = open(fname,'wb')
        pickle.dump(all_PROFS, fnout)
        fnout.close()

    del all_PROFS, PLATFORM_TYPE
    gc.collect()
