import numpy as np

# hist_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
#
# hist_z = np.zeros(shape=hist_bins.size)
#
# angle = 38
# idx = (np.abs(hist_bins - angle)).argmin()
# print(np.abs(hist_bins - angle))
#
# a,b = hist_bins.shape
# print(a,b)
# #---------------------------
a = np.array([45.50207415, 13.85894241,10.25124201, 14.49151809, 10.44445286,  8.5049243,
 12.64558588, 12.06353686,  9.90322278])
b = np.array([174,  36,  99, 698,  10,   7])

print(np.concatenate((a,b), axis=0))