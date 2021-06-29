import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

import ipdb

# from pylab import *
from matplotlib import dates

from datetime import datetime, timedelta
from datetime import date

import cmocean
import matplotlib.dates as mdates



# BBP = np.zeros(10)
# BBP[:] = np.nan


def plot_bbp(X, Y, BBP, XSTART, XEND, YMAX, WMO_dac, QC_Flags=None):

    if QC_Flags is None:
        QC_Flags = np.ones(BBP.shape)

    fig = plt.figure(figsize=(18,6))

    ax1 = fig.add_subplot(1,1,1)

    CMAP_BBP700 = cmocean.cm.thermal
    
    # create masked array of BBP    
#     BBP_masked = np.ma.masked_where(BBP==np.nan, BBP, copy=True)
    BBP_masked = np.ma.masked_where(QC_Flags!=1, BBP, copy=True)
    BBP_masked = np.ma.masked_where(np.isnan(BBP_masked), BBP_masked)
    BBP_masked = np.ma.filled(BBP_masked, np.nan)
    
    
    VMIN_BBP700 = np.nanpercentile(BBP_masked, 5)
    VMAX_BBP700 = np.nanpercentile(BBP_masked, 95)

    
    ax1.scatter(X, Y, s=64, c=BBP_masked, marker='o', edgecolors='none', vmin=VMIN_BBP700, vmax=VMAX_BBP700, cmap=CMAP_BBP700)

    ax1.set_ylim([0, YMAX])

    ax1.invert_yaxis()

    #####   set ticks and  labels
    days = mdates.DayLocator(interval=2)
    months = mdates.MonthLocator(interval=1)
    years = mdates.YearLocator()

    datemin = datetime.strptime(XSTART, '%Y-%m-%d').date()
    datemin = datemin.toordinal()
    datemax = datetime.strptime(XEND, '%Y-%m-%d').date()
    datemax = datemax.toordinal()

    ax1.set_xlim(datemin, datemax)


    fmt = mdates.DateFormatter('%m')

    ax1.xaxis.set_minor_locator(months)
    ax1.xaxis.set_major_locator(years)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_formatter(fmt)

    ax1.tick_params(axis='x', which='major', direction='out', pad=25, labelsize=20)
    ax1.tick_params(axis='x', which='minor', direction='out', pad=0.5, labelsize=20)
    ax1.tick_params(axis='both', which='both', labelsize=10)
    ax1.tick_params(axis='y', which='both', direction='out', labelsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    ax1.set_title("$b_{bp}$  " + WMO_dac, fontsize=30)

    ax1.set_ylabel('Pressure [dbars]', fontsize=20)


    ###prepare colorbar
    ax100 = fig.add_axes([0.92, 0.12, 0.025, 0.75])
    NORM = mpl.colors.Normalize(vmin=VMIN_BBP700, vmax=VMAX_BBP700)
    cb1 = mpl.colorbar.ColorbarBase(ax100, cmap=CMAP_BBP700, 
                       norm=NORM,
                       orientation='vertical')
    imaxes = plt.gca()
    plt.axes(cb1.ax)
    plt.yticks(fontsize=20)
    plt.axes(imaxes)
#     plt.close(fig)

    return


def plot_flags(X, Y, FLAGS, XSTART, XEND, YMAX, WMO_dac):
    
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl
#     CMAP = plt.get_cmap('Set1', 9)

    CMAP = cmocean.cm.thermal
    
    # Define the colors we want to use
    qc0 = np.array([100/256, 100/256, 100/256, 1]) # no QC
    qc1 = np.array([ 11/256,  11/256,  11/256, 1]) # good 
    qc2 = np.array([  0/256, 189/256,  10/256, 1]) # probably good
    qc3 = np.array([255/256,  10/256, 210/256, 1]) # probably bad
    qc4 = np.array([255/256,  10/256,  10/256, 1]) # bad
    qc5 = qc0
    qc6 = qc0
    qc7 = qc0
    qc8 = np.array([255/256, 247/256,   0/256, 1]) # interpolated value

    mapping = np.linspace(0,8,9)
    newcolors = np.empty((9, 4))
    newcolors[mapping == 0] = qc0
    newcolors[mapping == 1] = qc1
    newcolors[mapping == 2] = qc2
    newcolors[mapping == 3] = qc3
    newcolors[mapping == 4] = qc4
    newcolors[mapping == 5] = qc5
    newcolors[mapping == 6] = qc6
    newcolors[mapping == 7] = qc7
    newcolors[mapping == 7] = qc7
    newcolors[mapping == 8] = qc8

    CMAP = ListedColormap(newcolors)

    
    plt.clf()
    fig = plt.figure(figsize=(18,6))

#     print(np.any(FLAGS==0)) 
    
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_facecolor('k') # set background colour to black

    igt1 = np.where(FLAGS>1)
    ax1.scatter(X, Y, s=1, c=FLAGS, vmin=0, vmax=8, cmap=CMAP)
    ax1.scatter(X[igt1], Y[igt1], s=20, c=FLAGS[igt1], vmin=0, vmax=8, cmap=CMAP)

#     ax1.scatter(X, Y, s=1, c=FLAGS, marker='o', edgecolors='none')#, cmap = plt.get_cmap('Set1')) #, cmap=CMAP_BBP700)

#     igt1 = np.where(FLAGS>1)
#     print(FLAGS[igt1])
#     ax1.plot(X[igt1], Y[igt1])#, s=10, c='k', marker='o', edgecolors='none')#, cmap = plt.get_cmap('Set1')) #, cmap=CMAP_BBP700)

    ax1.set_ylim([-1, YMAX])
    ax1.invert_yaxis()

    
    ###prepare colorbar
    ax100 = fig.add_axes([0.92, 0.1, 0.025, 0.75])

    nnn = mpl.colors.Normalize(vmin=0, vmax=8)
    cb1 = mpl.colorbar.ColorbarBase(ax100, cmap=CMAP, orientation='vertical', norm=nnn)
    cb1.set_label('QC flags', fontsize=30)
    
    imaxes = plt.gca()
    plt.axes(cb1.ax)
    plt.yticks(fontsize=20)
    plt.axes(imaxes)
    
    
    
    
    
    #####   set ticks and  labels
    days = mdates.DayLocator(interval=2)
    months = mdates.MonthLocator(interval=1)
    years = mdates.YearLocator()

    datemin = datetime.strptime(XSTART, '%Y-%m-%d').date()
    datemin = datemin.toordinal()
    datemax = datetime.strptime(XEND, '%Y-%m-%d').date()
    datemax = datemax.toordinal()

    ax1.set_xlim(datemin, datemax)


    fmt = mdates.DateFormatter('%m')

    ax1.xaxis.set_minor_locator(months)
    ax1.xaxis.set_major_locator(years)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_formatter(fmt)

    ax1.tick_params(axis='x', which='major', direction='out', pad=25, labelsize=20)
    ax1.tick_params(axis='x', which='minor', direction='out', pad=0.5, labelsize=20)
    ax1.tick_params(axis='both', which='both', labelsize=10)
    ax1.tick_params(axis='y', which='both', direction='out', labelsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    ax1.set_title(" " + WMO_dac, fontsize=30)

    ax1.set_ylabel('Pressure [dbars]', fontsize=20)


#     plt.close(fig)


# from numpy import nan, median, isnan, nanmedian, nanmin, nanmax, where

# medfilt1 function similar to Octave's that does not bias extremes of dataset towards zero
def medfilt1(data, kernel_size, endcorrection='shrinkkernel'):
    """One-dimensional median filter"""
    halfkernel = int(kernel_size/2)
    data = np.asarray(data)

    filtered_data = np.empty(data.shape)
    filtered_data[:] = np.nan
#     innan = where(~isnan(data))
    
    for n in range(len(data)):
        i1 = np.nanmax([0, n-halfkernel])
        i2 = np.nanmin([len(data), n+halfkernel+1])
        filtered_data[n] = np.nanmedian(data[i1:i2])
        
    return filtered_data