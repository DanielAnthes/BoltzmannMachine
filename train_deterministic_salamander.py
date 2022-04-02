import matplotlib.pyplot as plt
import numpy as np
from bm import fit_bm_deterministic, clamped_statistics

# EXPERIMENT PARAMETERS
n_experiments = 5
exp_lls = list()

# generate all states
nneurons = 10  
allstates = [str(bin(x))[2:].zfill(nneurons) for x in list(range(2**nneurons))]
allstates = [[1 if x == '1' else -1 for x in y] for y in allstates]
allstates = np.array(allstates, dtype=np.float64)

# load salamander data

data = np.loadtxt("bint.txt")
data = data[:nneurons,:953]  # select first n neurons, select only one trial
data[data<.5] = -1  # convert to -1,1 format
data = data.T  # transpose to fit erxpected data shape
data = np.ascontiguousarray(data)  # contiguous in memory for performance

# fit boltzmann distribution

### HYPERPARAMETERS
lr = 0.001  # initial learning rate
schedule = 10000  # schedule, larger is slower
init_scale = 0.1  # weights and biases are initialized normally distributed, scale of gaussian
###

s_i_clamp, s_ij_clamp = clamped_statistics(data)
s_i_clamp = np.expand_dims(s_i_clamp, axis=1)

for i in range(n_experiments):
    print(f"\nRun: {i}\n")
    weights, theta, lls = fit_bm_deterministic(data, allstates, s_i_clamp, s_ij_clamp, lr, schedule, init_scale)
    exp_lls.append(lls)


plt.figure()
for l in exp_lls:
    plt.plot(range(len(l)), l)
plt.xlabel("Iteration")
plt.ylabel("data log likelihood")
plt.savefig("bm_explicit_salamander_convergence.png")

