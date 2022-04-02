import matplotlib.pyplot as plt
import numpy as np
from bm import fit_bm_mean_field, fit_bm_stochastic, clamped_statistics

n_runs = 2
lls_mf = list()
lls_stoc = list()

# generate all states
nneurons = 160
explicit = False

if explicit:
    allstates = [str(bin(x))[2:].zfill(nneurons) for x in list(range(2**nneurons))]
    allstates = [[1 if x == '1' else -1 for x in y] for y in allstates]
    allstates = np.array(allstates, dtype=np.float64)
else:
    allstates = np.zeros((nneurons,1))
# load salamander data

data = np.loadtxt("bint.txt")
data = data[:nneurons,:953]  # select first n neurons, select only one trial
data[data<-5] = -1  # convert to -1,1 format
data = data.T  # transpose to fit erxpected data shape
data = np.ascontiguousarray(data)  # contiguous in memory for performance

print(f"Dataset: Salamander data with {data.shape[1]} neurons and {data.shape[0]} samples")

### HYPERPARAMETERS
lr = 1e-3
schedule = 1000
init_scale = .01
n_glauber = 1000



s_i_clamp, s_ij_clamp = clamped_statistics(data)
s_i_clamp = np.expand_dims(s_i_clamp, axis=1)

for i in range(n_runs):
    print(f"\nrun: {i}")
    print("... mean field")
    weights, theta, lls = fit_bm_mean_field(data, allstates, s_i_clamp, s_ij_clamp, lr, schedule, init_scale, compare_explicit=False, explicit_Z=False)
    lls_mf.append(lls)
    print("\n... metropolis hastings")
    weights, theta, lls = fit_bm_stochastic(data, allstates, s_i_clamp, s_ij_clamp, lr, n_glauber, schedule, init_scale, compare_explicit=False, explicit_Z=False)
    lls_stoc.append(lls)

plt.figure()
for i in range(n_runs):
    plt.plot(range(len(lls_mf[i])), lls_mf[i], color='navy', alpha=.8)
    plt.plot(range(len(lls_stoc[i])), lls_stoc[i], color='darkorange', alpha=.8)
plt.savefig("comparison.png")
plt.show()
