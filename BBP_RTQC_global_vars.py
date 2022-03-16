# these are the tests and their codes
tests = {"A": "Negative (<5 dbar)",
         "A2": "Negative (>=5 dbar)",
         "B": "Noisy Profile",
         "C": "High-Deep Value",
         "E": "Missing Data",
         "G": "Parking Hook"
        }


# Global range
A_MIN_BBP700 = 0    # [1/m]
#A_MAX_BBP700 = 0.01 # [1/m] REVISED VALUE (very conservative estimate based on histograms in fig 2 of Bisson et al., 2019, 10.1364/OE.27.030191)
#A_MAX_BBP700 = 0.03 # [1/m] REVISED VALUE (very conservative estimate based on histograms in fig 2 of Bisson et al., 2019, 10.1364/OE.27.030191)

# Noisy profile
B_RES_THRESHOLD = 0.0005                    # [1/m] threshold for relative residuals
B_FRACTION_OF_PROFILE_THAT_IS_OUTLIER = 0.1 # fraction of profile with relative residuals above RES_THRESHOLD
B_PRES_THRESH = 100                         # [dbar] this is to avoid flagging profiles with spikes in surface data (likely good data)

# High-Deep Value
C_DEPTH_THRESH = 700          # [dbar] pressure threshold below which the test acts
C_DEEP_BBP700_THRESH = 0.0005 # [1/m] threshold for bbp at depth
C_N_of_ANOM_POINTS = 5        # number of anomalous points required for the test to fail

# Parking hook
G_DELTAPRES1 = 50  # [dbar] difference in PRES from parking pressure over which the test is implemented
G_DELTAPRES2 = 20  # [dbar] difference in PRES from parking pressure use to compute test baseline
G_DEV = 0.0002     # [1/m] deviation from baseline that identifies anomalous data points
G_DELTAPRES0 = 100 # [dbar] define how close PARK_PRES has to be to max(PRES)
                   #        for the test to proceed

# Missing Data
E_MIN_N_PERBIN = 1 # [-] minimum number of data points per bin
E_MAXPRES = 1000   # [dbar] pressure below which the profile is considered shallow

