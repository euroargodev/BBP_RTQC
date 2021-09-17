# Global range
A_MIN_BBP700 = 0 # [1/m]
A_MAX_BBP700 = 0.01 # [1/m] REVISED VALUE (very conservative estimate based on histograms in fig 2 of Bisson et al., 2019, 10.1364/OE.27.030191)

# Surface Hook
D_MIN_BBP700 = 0 # [1/m]
D_ISURF = 5 # [dbars] pressure threshold above which to check for negative values

# Parking hook
G_DELTAPRES1 = 50 # [dbars] difference in PRES from parking pressure over which the test is implemented
G_DELTAPRES2 = 20 # [dbars] difference in PRES from parking pressure use to compute test baseline
G_STDFACTOR = 3 # factor that multiplies the standard deviation to set the baseline

# Noisy profile
# B_RES_THRESHOLD = 0.001 # [1/m] threshold for relative residuals
# B_FRACTION_OF_PROFILE_THAT_IS_OUTLIER = 0.15 # fraction of profile with relative residuals above REL_RES_THRESHOLD
B_RES_THRESHOLD = 0.0005 # [1/m] threshold for relative residuals
B_FRACTION_OF_PROFILE_THAT_IS_OUTLIER = 0.1 # fraction of profile with relative residuals above REL_RES_THRESHOLD
B_PRES_THRESH = 100 # [dbars] # this is to avoid flagging profiles with spikes in surface data (likely good data)


C_DEPTH_THRESH = 800 #[dbars] below this threshold we consider it "deep"
C_DEEP_BBP700_THRESH = 0.0005 # [1/m] threshold for bbp at depth
C_N_of_ANOM_POINTS = 5 # number of anomalous points required for the test to fail

E_PRESTHRESH = 200 # [dbars] pressure below which the shallow-high-deep-value is computed
E_DEEP_BBP700_THRESH = C_DEEP_BBP700_THRESH


