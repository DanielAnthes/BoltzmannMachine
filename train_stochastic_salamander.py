import matplotlib.pyplot as plt
import numpy as np
from bm import fit_bm_stochastic, clamped_statistics

# generate all states
nneurons = 10  
allstates = [str(bin(x))[2:].zfill(nneurons) for x in list(range(2**nneurons))]
allstates = [[1 if x == '1' else -1 for x in y] for y in allstates]
allstates = np.array(allstates, dtype=np.float64)

# load salamander data

data = np.loadtxt("bint.txt")
data = data[:nneurons,:953]  # select first n neurons, select only one trial
data[data<-5] = -1  # convert to -1,1 format
data = data.T  # transpose to fit erxpected data shape
data = np.ascontiguousarray(data)  # contiguous in memory for performance

lr = 1e-3
s_i_clamp, s_ij_clamp = clamped_statistics(data)
s_i_clamp = np.expand_dims(s_i_clamp, axis=1)
weights, theta, lls = fit_bm_stochastic(data, allstates, s_i_clamp, s_ij_clamp, lr, nglauber=10000)

plt.figure()
plt.plot(range(len(lls)), lls)
plt.show()
