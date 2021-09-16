import numpy as np
import xarray as xr
import scipy.interpolate as spint
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



warnings.filterwarnings('ignore')

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
# test parameters defined outside function to make them global variables
def BBP_Global_range_test(BBP, BBPmf1, PRES, QC_Flags, QC_1st_failed_test, fn, VERBOSE=False, PLOT=False, SAVEPLOT=False):
    # BBP: nparray with all BBP data
    # BBPmf1: median-filtered BBP data
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    #
    # WHAT IS DONE: Then this tests fails, only the failing points are flagged (QC=3)

    # ### GLOBAL RANGE TEST (test order code "A")
    # <br>
    # #### Objective:
    # To detect and flag values of BBP532 and BBP700 that are outside the expected range.
    # <br>
    # <br>
    # #### What is done:
    # Check that:<br>
    # <code> medfilt1(BBP700) </code>  is in range [0, 0.001] m$^{-1} $.<br>
    # <code> medfilt1(BBP532) </code> is in range [0, 0.001] m$^{-1} $.<br>
    # <br>
    # The value of <code>A_MAX_BBP700=0.001</code> is taken as a very conservative estimate based on histograms in fig 2 of Bisson et al., 2019, 10.1364/OE.27.030191
    # <br>
    # <br>
    # #### QC flag if test fails
    # 3
    # <br>
    # Note: the entire profile is flagged, if any negative point is found (because if there is a negative value, then
    # it is worth looking at the profile during DMQC).
    # <br>
    # <br>
    # EXAMPLE (only positive values): csiro_BR5905397_080.nc, coriolis_BR7900591_240.nc<br>
    # EXAMPLE (medfilt1 negative values) coriolis_BR6903093_044.nc:
    # __________________________________________________________________________________________
    #



    FORCE_PLOT = False # this is to plot even if the test does not fails

    QC = 3
    QC_TEST_CODE = 'A' # or 'A2' if negative medfilt1 value is found   

    ISBAD = np.zeros(len(BBPmf1), dtype=bool) # initialise array with indices of where test failed  

    # this is the test 
    ibad = np.where( (BBPmf1 > A_MAX_BBP700) | (BBPmf1 < A_MIN_BBP700) )[0]

    ISBAD[ibad] = 1
    if np.any(ISBAD==1): # If ISBAD is not empty
        # flag entire profile if any negative value is found
        if np.any(BBPmf1 < A_MIN_BBP700):
            QC_TEST_CODE = 'A2'  
            ISBAD = np.where(BBPmf1)

        # apply flag
        QC_Flags[ISBAD] = QC
        QC_1st_failed_test[ISBAD] = QC_TEST_CODE


        if VERBOSE:
            print('Failed Global_Range_test')
            print('applying QC=' + str(QC) + '...')
            
        if PLOT:
            plot_failed_QC_test(BBP, BBPmf1, PRES, ISBAD, QC_Flags, QC_1st_failed_test, QC_TEST_CODE, fn, VERBOSE, FORCE_PLOT, SAVEPLOT)
        
    return QC_Flags, QC_1st_failed_test


##################################################################
##################################################################
# test parameters defined outside function to make them global variables
## D_MIN_BBP700 = 0 # [1/m]
## D_ISURF = 5 # [dbars] pressure threshold above which to check for negative values
def BBP_Surface_hook_test(BBP, BBPmf1, PRES, QC_Flags, QC_1st_failed_test, fn, VERBOSE=False, PLOT=False):
    # BBP: nparray with all BBP data
    # BBPmf1: median-filtered BBP data
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    #
    # WHAT IS DONE: Then this tests fails, only the failing points are flagged (QC=3)

    # ### SURFACE HOOK TEST (test order code "D")
    # <br>
    # #### Objective:
    # To detect and flag values of BBP532 and BBP700 that are negative near the surface.
    # <br>
    # <br>
    # #### What is done:
    # Check that:<br>
    # <code> BBP700 </code>  is greater than 0 m$^{-1}$ in the top 5 dbars.<br>
    # <code> BBP532 </code>  is greater than 0 m$^{-1}$ in the top 5 dbars.<br>
    # <br>
    # This test is needed because the medfilt1(BBP) used in the Global Range Test remove these few negative values near the surface.
    # This likely needed because of some miscalibration in the pressure that cause the BBP meter to collect data outside of the water.
    # <br>
    # <br>
    # #### QC flag if test fails
    # 3
    # <br>
    # <br>
    # EXAMPLE: coriolis_BD6901482_097.nc
    # __________________________________________________________________________________________
    #

    FORCE_PLOT = False # this is to plot even if the test does not fails
    
    QC = 3
    QC_TEST_CODE = 'D'    
    ISBAD = np.zeros(len(BBPmf1), dtype=bool) # initialise flag
    iSURF  = np.where(PRES<=D_ISURF)[0]
     
    # this is the test 
    ibad = np.where(  (BBP[iSURF] < D_MIN_BBP700) )[0]
    ISBAD[ibad] = 1
    if np.any(ISBAD==1): # If ISBAD is not empty
        # apply flag
        QC_Flags[ISBAD] = QC
        QC_1st_failed_test[ISBAD] = QC_TEST_CODE
        
        if VERBOSE:
            print('Failed Surface_hook_test')
            print('applying QC=' + str(QC) + '...')
            
        if PLOT:
            plot_failed_QC_test(BBP, BBPmf1, PRES, ISBAD, QC_Flags, QC_1st_failed_test, QC_TEST_CODE, fn, VERBOSE, FORCE_PLOT)
        
    return QC_Flags, QC_1st_failed_test


##################################################################
##################################################################
# test parameters defined outside function to make them global variables
#G_DELTAPRES1 = 50 # [dbars] difference in PRES from parking pressure over which the test is implemented
#G_DELTAPRES2 = 20 # [dbars] difference in PRES from parking pressure use to compute test baseline
#G_STDFACTOR = 3 # factor that multiplies the standard deviation to set the baseline
def BBP_Parking_hook_test(BBP, BBPmf1, PRES, maxPRES, PARK_PRES, QC_Flags, QC_1st_failed_test, 
                          fn, VERBOSE=False, PLOT=False):
    # BBP: nparray with all BBP data
    # BBPmf1: nparray with medfilt BBP data
    # maxPRES: maximum pressure recorded in this profile
    # PARK_PRES: programmed parking pressure for this profile
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    #
    # WHAT IS DONE: Then this tests fails, only the failing points are flagged (QC=3)

    # ### PARKING HOOK TEST (test order code "G")
    # <br>
    # #### Objective:
    # To detect and flag values of BBP532 and BBP700 that anomalously high at the start (i.e., bottom) of the profile, when the parking PRES is close to the maximum recorded PRES.
    # <br>
    # <br>
    # #### What is done:<br>
    # Compute <code>baseline</code> using BBPmf1 above which the test is triggered using data that are in an PRES interval <code>iPRESmed</code> between 50 (G_DELTAPRES1) and 20 (G_DELTAPRES1) dbars above the maxPRES.
    # The <code>baseline</code> is defined as <code>median(BBPmf1[iPREDmed]) + G_STDFACTOR*robstd(BBPmf1[iPREDmed])</code>, where <code>robstd(BBPmf1[iPREDmed])</code> is the robust standard deviation.<br>
    # The test checks that:
    # <code>BBPmf1 > baseline</code> <br>
    # <br>
    # <br>
    # #### QC flag if test fails
    # 4
    # <br>
    # <br>
    # EXAMPLE: coriolis_BD6901580_107.nc
    # __________________________________________________________________________________________
    #

    FORCE_PLOT = False # this is to plot even if the test does not fails
    
    QC = 4
    QC_TEST_CODE = 'G'    
    ISBAD = np.zeros(len(BBP), dtype=bool) # initialise flag
        
    if (np.isnan(maxPRES)) | (np.isnan(PARK_PRES)):
        print('WARNING:\nmaxPRES='+str(maxPRES)+' dbars,\nPARK_PRES='+str(PARK_PRES))
        ipdb.set_trace()
        
     
    # check that there are enough data to run the test
    imaxPRES = np.where(PRES==maxPRES)[0][0]
    deltaPRES = PRES[imaxPRES] - PRES[imaxPRES-1]
    if deltaPRES>G_DELTAPRES2:
        print('vertical resolution is too low to check for Parking Hook')
        return QC_Flags, QC_1st_failed_test
        
        
    # check if max PRES is 'close' (i.e., within 100 dbars) to PARK_PRES
    if abs(maxPRES - PARK_PRES)>=100:
        return QC_Flags, QC_1st_failed_test


    # define PRES range over which to compute the baseline for the test
    iPRESmed = np.where((PRES>= maxPRES - G_DELTAPRES1 ) & (PRES< maxPRES - G_DELTAPRES2) )[0]
    # define PRES range over which to apply the test
    iPREStest = np.where((PRES>= maxPRES - G_DELTAPRES1 ))[0]

    # compute parameters to define baseline above which test fails
    medBBP = np.nanmedian(BBP[iPRESmed])
    stdBBP = np.nanstd(BBP[iPRESmed])
#     stdBBP = (np.nanpercentile(BBPmf1[iPRESmed], 84) - np.nanpercentile(BBPmf1[iPRESmed], 16))/2.
    baseline = medBBP + G_STDFACTOR*stdBBP
    
    # this is the test
    ibad = np.where( BBP[iPREStest] > baseline )[0]
    ISBAD[iPREStest[ibad]] = 1  
    
    if np.any(ISBAD==1): # If ISBAD is not empty
        # apply flag
        QC_Flags[ISBAD] = QC
        QC_1st_failed_test[ISBAD] = QC_TEST_CODE
        
        if VERBOSE:
            print('Failed Parking_hook_test')
            print('applying QC=' + str(QC) + '...')
            
        if PLOT:
            plot_failed_QC_test(BBP, BBP, PRES, ISBAD, QC_Flags, QC_1st_failed_test, QC_TEST_CODE, fn, VERBOSE, FORCE_PLOT)
        
    return QC_Flags, QC_1st_failed_test


# ## Tests that flag the entire profile


##################################################################
##################################################################
# test parameters defined outside function to make them global variables
## D_MIN_BBP700 = 0 # [1/m]
## D_ISURF = 5 # [dbars] pressure threshold above which to check for negative values
## example csiro/5905022/profiles/BD5905022_053.nc
def BBP_Negative_nonsurface_test(BBP, PRES, QC_Flags, QC_1st_failed_test, fn, VERBOSE=False, PLOT=False):
    # BBP: nparray with all BBP data
    # BBPmf1: median-filtered BBP data
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    #
    # WHAT IS DONE: Then this tests fails, only the failing points are flagged (QC=3)

    # ### NEGATIVE NON-SURFACE TEST (test order code "F")
    # <br>
    # #### Objective:
    # To detect and flag values of BBP532 and BBP700 that are negative below the surface.
    # <br>
    # <br>
    # #### What is done:
    # Check that:<br>
    # <code> BBP700 </code>  is less than 0 m$^{-1}$ below 5 dbars.<br>
    # <code> BBP532 </code>  is less than 0 m$^{-1}$ below 5 dbars.<br>
    # <br>
    # This test is needed because the medfilt1(BBP) used in the Global Range Test remove these few negative values near the surface.
    # <br>
    # <br>
    # #### QC flag if test fails
    # 3
    # <br>
    # <br>
    # EXAMPLE: coriolis_BD6901527_182.nc<br>
    # __________________________________________________________________________________________
    #

    FORCE_PLOT = False # this is to plot even if the test does not fails

    QC = 3
    QC_TEST_CODE = 'F'    
    ISBAD = np.zeros(len(BBP), dtype=bool) # initialise flag
#     ISBAD = np.array([])
#     iDEEP  = np.where( PRES > D_ISURF )[0]
     
    # this is the test 
    ibad = np.where( (BBP < D_MIN_BBP700) & (PRES > D_ISURF) )[0] 
#     ipdb.set_trace()
    if np.any(ibad):
        ISBAD[ibad] = 1
        iQChigher = np.where(QC_Flags < QC) 
        if iQChigher:
            if VERBOSE:
                print('Failed BBP_Negative_nonsurface_test')
                print('applying QC=' + str(QC) + '...')
                
            QC_Flags[iQChigher] = QC
            QC_1st_failed_test[iQChigher] = QC_TEST_CODE
            
        if ((np.any(ISBAD)) & (PLOT==True)) | (FORCE_PLOT==True):
            plot_failed_QC_test(BBP, BBP, PRES, ISBAD, QC_Flags, QC_1st_failed_test, QC_TEST_CODE, fn, VERBOSE, FORCE_PLOT)
        
    return QC_Flags, QC_1st_failed_test


##################################################################
##################################################################
#B_RES_THRESHOLD = 0.001 # [1/m] threshold for relative residuals
#B_FRACTION_OF_PROFILE_THAT_IS_OUTLIER = 0.15 # fraction of profile with relative residuals above REL_RES_THRESHOLD
def BBP_Noisy_Profile_test(BBP, BBPmf1, PRES, QC_Flags, QC_1st_failed_test, fn, VERBOSE=False, PLOT=True):
    # BBP: nparray with all BBP data
    # BBPmf1: smooth BBP array (medfilt1(BBP700, 31)
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    #
    # WHAT IS DONE: When the test fails, all points in the profile are flagged (QC=3)

    # ### BBP NOISY-PROFILE TEST  (test order code "B")
    # <br>
    # #### Objective:
    # To detect and flag profiles of BBP that are affected by noisy data.
    # <br><br>
    # #### What is done:
    # Compute <code> res = abs(BBP-medfilt1(BBP, 31))</code>.
    #
    # Flag profiles where at least <code>15%</code> of the profile has <code>res > 0.001</code>  m$^{-1}$.<br>
    # <br><br>
    # #### QC flag if test fails
    # 3
    # <br>
    # <br>
    # EXAMPLE: bodc_BR6901183_130.nc, aoml_BD5905108_073.nc
    # __________________________________________________________________________________________


    QC = 3; # flag to apply if the result of the test is true
    QC_TEST_CODE = 'B'
    ISBAD = np.array([]) # flag for noisy profile
    
    FORCE_PLOT = False # plot even if the test does not fail (used when developing the test)

#     rel_res = np.empty(BBP.shape)
#     rel_res[:] = np.nan
    
    res = np.empty(BBP.shape)
    res[:] = np.nan
    


    innan = np.where(~np.isnan(BBP))[0]
    
    if len(innan)>10: # if we have at least 10 points in the profile
        res[innan] = np.abs(BBP[innan]-BBPmf1[innan])
        ioutliers = np.where(abs(res)>B_RES_THRESHOLD)[0] # index of where the rel res are greater than the threshold

        if len(ioutliers)/len(innan)>=B_FRACTION_OF_PROFILE_THAT_IS_OUTLIER: # this is the actual test: are there more than a certain fraction of points that are noisy?
            ISBAD = ioutliers

    # update QC_Flags to 3 when bad profiles are found
    if len(ISBAD)>0:     
        iQChigher = np.where(QC_Flags < QC) 
        if iQChigher:
            if VERBOSE:
                print('Failed BBP_Noisy_Profile_test')
                print('applying QC=' + str(QC) + '...')
                
            QC_Flags[iQChigher] = QC
            QC_1st_failed_test[iQChigher] = QC_TEST_CODE

    if ((len(ISBAD)>0) & (PLOT==True)) | (FORCE_PLOT==True):
        plot_failed_QC_test(res, res*0., PRES, ISBAD, QC_Flags, QC_1st_failed_test, QC_TEST_CODE, fn, VERBOSE, FORCE_PLOT)

    
    return QC_Flags, QC_1st_failed_test, res


##################################################################
##################################################################
#C_DEPTH_THRESH = 800 #[dbars] below this threshold we consider it "deep"
#C_DEEP_BBP700_THRESH = 0.0005 # [1/m] threshold for bbp at depth
#C_N_of_ANOM_POINTS = 5 # number of anomalous points required for the test to fail
def BBP_High_Deep_Values_test(BBPmf1, PRES, QC_Flags, QC_1st_failed_test, fn, VERBOSE=False, PLOT=True):
    # BBP: nparray with all BBP data
    # BBPmf1: smooth BBP array (medfilt1(BBP700, 31)
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    #
    # WHAT IS DONE: When the test fails, all points in the profile are flagged (QC=3)

    # ### BBP HIGH-DEEP-VALUES TEST  (test order code "C")
    # <br>
    # #### Objective:
    # To detect and flag profiles of BBP that have at least some (5) points anomalously high values at depth. It could indicate multiple problems: wrong calibration coefficients, biofouling, bad sensor, grounding, etc.
    # <br><br>
    # #### What is done:
    # Check if <code>median(BBP700)</code> below <code>800 dbars</code> is above a threshold of <code>0.0005 </code> m$^{-1}$.  (this is half the value typical for surface bbp in the oligotrophic ocean, smoothed bbp data at depth are expected to be lower than this value).
    #
    # Flag entire profile.
    # <br><br>
    # #### QC flag if test fails
    # 3
    # <br>
    # <br>
    # EXAMPLE: coriolis_BR6902827_363.nc
    # __________________________________________________________________________________________

    FORCE_PLOT = False # this is to plot even if the test does not fails
        
    QC = 3; # flag to apply if the result of the test is true
    QC_TEST_CODE = 'C'
    ISBAD = np.zeros(len(BBPmf1), dtype=bool) # flag for noisy profile
    
    # this is the test 
    iDEEP = np.where(PRES>C_DEPTH_THRESH)
    if (np.nanmedian(BBPmf1[iDEEP]) > C_DEEP_BBP700_THRESH) & ( len(BBPmf1[iDEEP]) >= C_N_of_ANOM_POINTS):
        ISBAD = np.ones(len(BBPmf1), dtype=bool)
    
    if np.any(ISBAD==1): # if ISBAD, then apply QC_flag=3  
        iQChigher = np.where(QC_Flags < QC) # but first check that there are no QCflags > than the one we want to assign in this profile
        if iQChigher:
            if VERBOSE:
                print('Failed High_Deep_Values_test')
                print('applying QC=' + str(QC) + '...')

                
            QC_Flags[iQChigher] = QC
            QC_1st_failed_test[iQChigher] = QC_TEST_CODE

            if PLOT:
                plot_failed_QC_test(BBPmf1, BBPmf1, PRES, ISBAD, QC_Flags, QC_1st_failed_test, QC_TEST_CODE, fn, VERBOSE, FORCE_PLOT)


 
    return QC_Flags, QC_1st_failed_test


##################################################################
##################################################################
#E_PRESTHRESH = 200 # [dbars] pressure below which the shallow-high-deep-value is computed
#E_DEEP_BBP700_THRESH = C_DEEP_BBP700_THRESH
def BBP_Missing_Data_test(BBP, PRES, QC_Flags, QC_1st_failed_test, fn, VERBOSE=False, PLOT=True):
    # BBP: nparray with all BBP data
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # fn: name of corresponding B-file
    #
    # WHAT IS DONE: When the test fails, all points in the profile are flagged (QC=2)

    # ### BBP MISSING-DATA TEST  (test order code "E")
    # <br>
    # #### Objective:
    # To detect and flag profiles of BBP that have a large fraction of missing data. This test can also detect profiles that are too shallow with respect to the Argo mission.
    # <br>
    # <br>
    # #### What is done:
    # Ensure that we have at least 1 measurements every 100 dbars in the upper 1000 dbars.
    #
    # Flag entire profile.
    # <br><br>
    # #### QC flag if test fails
    # 2 if shallow profile with BBP(PRES>200dbars)<=E_DEEP_BBP700_THRESH<br>
    # 3 if shallow profile with BBP(PRES>200dbars)>E_DEEP_BBP700_THRESH<br>
    # 3 if shallow profile with maxPRES<200 dbars<br>
    # 3 if data are available in less than 10 bins in the upper 1000 dbars<br>
    # 4 if only data within only one bin in the upper 1000 dbars
    # <br>
    # <br>
    # EXAMPLE (shallow profilewith BBP(PRES>200dbars)<E_DEEP_BBP700_THRESH, QC=2): coriolis_BR6902827_005.nc, csiro_BD1901338_660<br>
    # EXAMPLE (shallow profile with BBP(PRES>200dbars)>E_DEEP_BBP700_THRESH, QC=3): csiro_BD1901338_006.nc<br>
    # EXAMPLE (shallow profile with maxPRES<200 dbars, QC=3): csiro_BD1901338_032.nc<br>
    # EXAMPLE (data in less than 10 bins, QC=3): coriolis_BR7900560_060.nc<br>
    # EXAMPLE (data in only one bin, QC=4): coriolis_BD7900591_077.nc<br>
    # __________________________________________________________________________________________

    FORCE_PLOT = False # this is to plot even if the test does not fails
        
    QC_all = [np.nan, np.nan, np.nan, np.nan]    
    QC_all[0] = 2 # flag to apply if shallow profile 
    QC_all[1] = 3 # flag to apply if the result of the test is true
    QC_all[2] = 4 # flag to apply if there are data only within one size bin
    
    QC_TEST_CODE = 'E'
    MIN_N_PERBIN = 1 # minimum number of points in each bin
    ISBAD = 0 # flag for noisy profile
    
    # bin the profile into 100-dbars bins
    bins = np.linspace(50, 1000, 10) # create 10 bins between 0 and 1000 dbars
    PRESbin = np.digitize(PRES, bins) # assign PRES values to each bin
    bin_counts = np.zeros(bins.shape)*np.nan
    for i in range(len(bins)):
        if i==0:
            bin_counts[i] = len(np.where(PRES<bins[i])[0])
        else:
            bin_counts[i] = len(np.where((PRES>=bins[i-1]) & (PRES<bins[i]))[0])
            
    # this is the actual test
    if np.any(np.nonzero(bin_counts<MIN_N_PERBIN)[0]):
        ISBAD = 1  
        
        # find which bins contain data
        nonempty = np.where(bin_counts>0)[0] # index of bins with data inside
        
        if len(nonempty)>1: ########## TRY THIS INSTEAD LATER  if nonempty.any():
            test_bin = np.linspace(0, nonempty[-1], nonempty[-1]+1) # create array with consecutive indices from 0 to the last element of nonempty
            
            # if there is only one bin with data then
            if len(np.nonzero(bin_counts>MIN_N_PERBIN)[0])==1: 
                print("data only in one bin: QC=" + str(QC_all[2]))
                QC = QC_all[2]

            # if there are consecutive bins from zero index 
            # and if not all bins contain data
            elif (np.all(test_bin==nonempty)) & (nonempty[-1]<9) & (np.nanmax(PRES) >= bins.max()): 
                print("shallow profile due to missing data: QC=" + str(QC_all[1]))
                QC = QC_all[1]
                
            # check if max(PRES)<maxPresbin to decide if this was a profile that was programmed to be shallow
            elif (np.all(test_bin==nonempty)) & (nonempty[-1]<9) & (np.nanmax(PRES) < bins.max()): 
                print("shallow profile (maxPRES="+str(np.nanmax(PRES))+") dbars. Need more checks...") 

                # compute median value below 200 dbars to check for high-deep values
                iGT200 = np.where(PRES>E_PRESTHRESH)[0]
        
                if not iGT200.any(): # if there are no data deeper than 200 dbars, then set QC=3 without checking what the values are
                    print("----profile shallower than 200 dbars: QC=" + str(QC_all[1]))
                    QC = QC_all[1]

                else:    
                    # compute median BBP value below 200 dbars
                    medBBPGT200 = np.nanmedian(BBP[iGT200])

                    # if there are no non-NaN BBP values
                    if (not medBBPGT200) | np.isnan(medBBPGT200):
                         print("----shallow profile (maxPRES="+str(np.nanmax(PRES))+" dbars) no BBP data below 200 dbars) : QC=" + str(QC_all[1]))
                         QC = QC_all[1]

                    elif medBBPGT200<E_DEEP_BBP700_THRESH:    
                    # set ISBAD so that the test does not fail when it's a shallow profile without high-deep values
                        print("----shallow profile (maxPRES="+str(np.nanmax(PRES))+" dbars)) : QC=" + str(QC_all[0]))
                        ISBAD = 0
                        QC = QC_all[0]

                    elif medBBPGT200>=E_DEEP_BBP700_THRESH:
                        print("----shallow profile (maxPRES="+str(np.nanmax(PRES))+" dbars) with high deep values (medBBPGT200=" + str(medBBPGT200) + "): QC=" + str(QC_all[1]))
                        QC = QC_all[1]
                    

            # if missing data in the profile, but not cosecutively from bottom, then  
            else:
                print("data in some bins missing: QC=" + str(QC_all[1]))
                QC = QC_all[1]
                
        else: # this is for when we have no data at all, then
            print("no data at all: QC=4")
            QC = QC_all[2]


         
    if ISBAD==1: # if ISBAD, then apply QC_flag          
        iQChigher = np.where(QC_Flags < QC) # but first check that there are no QCflags > than the one we want to assign in this profile
        if iQChigher:
            if VERBOSE:
                print('Failed Missing_Data_test')
                print('applying QC=' + str(QC) + '...')

                
            QC_Flags[iQChigher] = QC
            QC_1st_failed_test[iQChigher] = QC_TEST_CODE

            if PLOT:
                plot_failed_QC_test(BBP, bin_counts, PRES, ISBAD, QC_Flags, QC_1st_failed_test, QC_TEST_CODE, fn, VERBOSE, FORCE_PLOT)


    return QC_Flags, QC_1st_failed_test


##################################################################
##################################################################
# function to plot results of applying test to dataset
def plot_failed_QC_test(BBP, BBPmf1, PRES, ISBAD, QC_Flags, QC_1st_failed_test, QC_TEST_CODE, fn, VERBOSE=False, FORCE_PLOT=False, SAVEPLOT=False):
    # BBP: nparray with all BBP data
    # BBPmf1: median-filtered BBP data
    # ISBAD: index marking the data that failed the QC test
    # QC_Flags: array with QC flags
    # QC_flag_1st_failed_test: array with info on which test failed QC_TEST_CODE
    # QC_TEST_CODE: code of the failed test
    #FORCE_PLOT: flag to plot the test plot even if the test has not failed
    
    DIR_OUT = './plots/'

#     %matplotlib inline   
#     plt.ioff() # this is needed to avoid showing the plot (i.e. only saving it)
    
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(1,1,1)

    innan = np.nonzero(~np.isnan(BBP))
    
    
    # check that there are enough data to plot
    if len(BBP)<2:
        if VERBOSE:
            print("not enough data to plot... exiting")
        return
    

    if (not (QC_TEST_CODE == "0")) & (FORCE_PLOT == False): # this is to plot the original profile 
        ax1.plot(BBPmf1[ISBAD], PRES[ISBAD], 'o', ms=10, color='r', mfc='r', alpha=0.7)
        ax1.plot(BBP[innan], PRES[innan], 'o-', ms=3, color='k', mfc='none', alpha=0.7) # <<<<<<<<<<<<<<<<<<

    if QC_TEST_CODE!='E':
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
#         ax1.set_xscale('log')
        ax1.set_xlim((-0.001, 0.015))

    elif QC_TEST_CODE =='B':
        ax1.plot(B_RES_THRESHOLD*np.ones(2), [-5, 2000], '--', color='r', mfc='none', alpha=0.7)
        ax1.set_xlim( (1e-6, 1e-2 ) )
        ax1.set_xscale('log')
#         rstd = (np.percentile(rel_res, 84) - np.percentile(rel_res, 16)/2.)
#         ax1.text(-2, 100, str(rstd), fontsize=20)

    if QC_TEST_CODE == "E":
        bin_counts = BBPmf1+10.
        ax1.cla()
        bins = np.linspace(50, 1000, 10)
        ax1.barh(bins-50, bin_counts/2000., height=97, zorder=0 )
        ifail = np.where(bin_counts<10+1)
        ax1.barh(bins[ifail]-50, bin_counts[ifail]/2000., height=97, color='r', zorder=1 )
        ax1.plot(BBP, PRES, 'k.-', zorder=60)   
        
        
#         ax1.plot(BBP, PRES, 'o', ms=10, color='r', mfc='r', alpha=0.7)
    # ax1.set_ylim((np.nanpercentile(O2sat_surf,0.1), np.nanpercentile(O2sat_surf,99.1)))
    # ax1.plot(BBP700[ibad[1],:], Y[ibad[1],:], 'o', ms=2, color='r', mfc='none', alpha=0.7)
    
        
    ax1.grid('on')

    ax1.set_ylim([-5, 2000])    
    ax1.invert_yaxis()

    ax1.set_xlabel('BBP [1/m]', fontsize=20)
    ax1.set_ylabel('PRES [dbars]', fontsize=20)
    ax1.set_title('QC='+QC_TEST_CODE+" "+fn.split('/')[-1], fontsize=20, color='r', fontweight='bold')

    if SAVEPLOT:
        fname = DIR_OUT + "/" + fn.split('/')[-3] + "/" + fn.split('/')[-4] + "_" + fn.split('/')[-1] + "_" + QC_TEST_CODE+ ".png"
        fig.savefig(fname, dpi = 75) 

    # minimise memory leaks
    plt.close(fig)
    gc.collect()
    
    return



# function to extract basic data from Argo NetCDF meta file
def rd_WMOmeta(iwmo):
    # read meta file to extract info on PARKING DEPTH
    fn = glob.glob(MAIN_DIR + 'dac/' + iwmo + '/*meta.nc')[0]
    ds_config = xr.open_dataset(fn)


    ## extract info on SENSOR
    if not np.all(ds_config.SENSOR.astype('str').str.contains('BACKSCATTERINGMETER_BBP700')):
        print("----this float does not have SENSOR metadata")
        SENSOR_MODEL = 'no metadata'
        SENSOR_MAKER = 'no metadata'
        SENSOR_SERIAL_NO = 'no metadata'
    else:    
        iBBPsensor = np.where(ds_config.SENSOR.astype('str').str.contains('BACKSCATTERINGMETER_BBP700'))[0][0] # find index of BBP meter
        SENSOR_MODEL = ds_config.SENSOR_MODEL[iBBPsensor].astype('str').values
        SENSOR_MAKER = ds_config.SENSOR_MAKER[iBBPsensor].astype('str').values
        SENSOR_SERIAL_NO = ds_config.SENSOR_SERIAL_NO[iBBPsensor].astype('str').values
    
    if ds_config.PLATFORM_TYPE.values:
        PLATFORM_TYPE = str(ds_config.PLATFORM_TYPE.values.astype(str))
    else:
        PLATFORM_TYPE = 'no metadata'    
    print("PLATFORM_TYPE=" + PLATFORM_TYPE)
    
    # extract CONFIG_MISSION_NUMBER from meta file
    miss_no_float = ds_config.CONFIG_MISSION_NUMBER.values
    
    return ds_config, SENSOR_MODEL, SENSOR_MAKER, SENSOR_SERIAL_NO, PLATFORM_TYPE, miss_no_float



# read BBP and PRES
def rd_BBP(fn_p, miss_no_float, ds_config):
    ds = xr.open_dataset(fn_p)

    # check if BBP700 is present
    v = set(ds.data_vars)
    if 'BBP700' not in v:
        if VERBOSE:
            print('no BBP700 for this cycle')
        ds.close()
        return 


    # find N_PROF where the BBP700 data are stored     
    tmp_bbp = ds.BBP700.values # note that BBP700[N_PROF,N_LEVELS]
    tmp = [np.any(~np.isnan(tmp_bbp[i][:])) for i in range(tmp_bbp.shape[0])] # find which of the different columns of tmp_bbp has at least one non-NaN element
    if np.any(tmp):
        N_PROF = np.where(tmp)[0][0]
    else:
        if VERBOSE:
            print("this profile has less than 5 data points: skipping ")
        return


    BBP700 = ds.BBP700[N_PROF].values
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

    return PRES, BBP700, JULD, LAT, LON, BBP700mf1, miss_no_prof, PARK_PRES, maxPRES, innan




# function to apply tests and plot results (needed in function form for parallel processing)
def QC_wmo(iwmo, VERBOSE=True, SAVEPKL=False):
 
    PLOT = True

    print(iwmo)

    if len(iwmo)==0:
        return

    # read meta file
    [ds_config, SENSOR_MODEL, SENSOR_MAKER, SENSOR_SERIAL_NO, PLATFORM_TYPE, miss_no_float] = rd_WMOmeta(iwmo) 
    
    # list single profiles
    fn2glob = MAIN_DIR + "dac/" + iwmo + "/profiles/" + "B*" + iwmo.split("/")[-1] + "*_[0-9][0-9][0-9].nc"
    fn_single_profiles = np.sort(glob.glob(fn2glob))

    # create dir for output plots
    dout = DIR_PLOTS + "/" + fn_single_profiles[0].split('/')[-3]
    if not os.path.isdir(dout):
        os.mkdir(dout)
        if VERBOSE:
            print("created " + dout)
    else: # remove old plots + pkl file from this dir
        if VERBOSE:
            print("removing old plots in " + dout + "...")     
        oldfn = glob.glob(DIR_PLOTS + iwmo.split("/")[-1] + "/*.p*")
        print(DIR_PLOTS + iwmo.split("/")[-1] + "/*.p*")
        [os.remove(i) for i in oldfn]   
        if VERBOSE:
            print("...done")
        
    # initialise variables that will store all data from this float
    all_PRES = []
    all_BBP700 = []
    all_BBP700_QC_flag = []
    all_BBP700_1st_fail = []

    all_PROFS = []
    all_LAT = np.empty(len(fn_single_profiles))
    all_LON = np.empty(len(fn_single_profiles))
    all_JULD = np.empty(len(fn_single_profiles))

    for ifn_p, fn_p in enumerate(fn_single_profiles):
        if VERBOSE:
            print(fn_p)

        # read BBP and PRES + other data from profile file
        [ PRES, BBP700, JULD, LAT, LON, BBP700mf1, miss_no_prof, PARK_PRES, maxPRES, innan] = rd_BBP(fn_p, miss_no_float, ds_config)

        # initialise arrays with QC flags[0,:] = 1 (good data)
        BBP700_QC_flags = np.zeros(BBP700.shape)+1
        BBP700_QC_1st_failed_test = np.full(shape=BBP700.shape, fill_value='0')
        
        # Plot original profile even if no QC flag is raisef
        plot_failed_QC_test(BBP700, BBP700mf1, PRES, BBP700*np.nan, BBP700_QC_flags, BBP700_QC_1st_failed_test, '0', fn_p)
        
        # GLOBAL-RANGE TEST for BBP700
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Global_range_test(BBP700, BBP700mf1, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, VERBOSE, PLOT)

        # SURFACE-HOOK TEST for BBP700
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Surface_hook_test(BBP700, BBP700mf1, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, VERBOSE, PLOT)

        # PARKING-HOOK TEST for BBP700
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Parking_hook_test(BBP700, BBP700mf1, PRES, maxPRES, PARK_PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, VERBOSE, PLOT)

        # BBP_NOISY_PROFILE TEST
        BBP700_QC_flag, BBP700_QC_1st_failed_test, rel_res = BBP_Noisy_Profile_test(BBP700, BBP700mf1, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, VERBOSE, PLOT)

        # HIGH_DEEP_VALUES TEST
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_High_Deep_Values_test(BBP700mf1, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, VERBOSE, PLOT)
        
        # MISSING_DATA TEST
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Missing_Data_test(BBP700, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, VERBOSE, PLOT)
        
        # NEGATIVE NON-SURFACE TEST
        BBP700_QC_flag, BBP700_QC_1st_failed_test = BBP_Negative_nonsurface_test(BBP700, PRES, BBP700_QC_flags, BBP700_QC_1st_failed_test, fn_p, VERBOSE, PLOT)

        # ANIMAL SPIKES

        # PROFILE-STEP-CHANGE TEST (e.g., 6902737_262)
        
        # GROUNDED_BBP TEST
    
        # STUCK-VALUE TEST

        
        ideep = np.where((np.asarray(PRES)>950.) & (np.asarray(PRES)<1050.))[0]


        
        

        # save results in res list
        all_PRES.extend(PRES[innan])
        all_BBP700.extend(BBP700[innan])
        all_BBP700_QC_flag.extend(BBP700_QC_flag[innan])
        all_BBP700_1st_fail.extend(BBP700_QC_1st_failed_test[innan])
        

        all_LAT[ifn_p] = LAT.tolist()
        all_LON[ifn_p] = LON.tolist()
        all_JULD[ifn_p] = JULD
        
        
        # save results in dictonary
        prof = {"JULD": JULD,
                "LAT": LAT.tolist(),
                "LON": LON.tolist(),
                "PRES": PRES[innan],
                "BBP700": BBP700[innan],
                "BBP700_QC_flag": BBP700_QC_flag[innan],
                "BBP700_QC_1st_failed_test": BBP700_QC_1st_failed_test[innan]
               }
        
        all_PROFS.extend([prof])
        del prof
        
    # add META variables at the end
    all_PROFS.extend([{'PARK_PRES':PARK_PRES, 
                       'SENSOR_MODEL':SENSOR_MODEL, 
                       'SENSOR_MAKER':SENSOR_MAKER, 
                       'SENSOR_SERIAL_NO':SENSOR_SERIAL_NO, 
                       'PLATFORM_TYPE':PLATFORM_TYPE,
                       'iWMO': iwmo
                        }])
    
    if SAVEPKL:
        # save results in pickled file (https://www.datacamp.com/community/tutorials/pickle-python-tutorial)
        fname = DIR_OUT + fn_p.split('/')[-3] + "/" + fn_p.split('/')[-4] + "_" + fn_p.split('/')[-1].split('.')[0].split('_')[0] +  ".pkl"
        fnout = open(fname,'wb')
        pickle.dump(all_PROFS, fnout)
        fnout.close()

    del all_PROFS, PLATFORM_TYPE
    gc.collect()
    



