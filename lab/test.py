import numpy as np

hist_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])

hist_z = np.zeros(shape=hist_bins.size)

angle = 38
idx = (np.abs(hist_bins - angle)).argmin()
print(np.abs(hist_bins - angle))

a,b = hist_bins.shape
print(a,b)