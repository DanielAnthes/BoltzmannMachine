import matplotlib.pyplot as plt
import numpy as np
from bm import fit_bm_mean_field, clamped_statistics


# EXPERIMENT PARAMETERS
n_experiments = 5
exp_lls = list()

# generate all states
nneurons = 10  
allstates = [str(bin(x))[2:].zfill(nneurons) for x in list(range(2**nneurons))]
allstates = [[1 if x == '1' else -1 for x in y] for y in allstates]
allstates = np.array(allstates, dtype=np.float64)

# generate dataset
ndata = 500
idx = np.random.choice(np.arange(2**10), replace=True, size=ndata)
data = allstates[idx]

lr = 1e-3
schedule = 1000  # schedule, larger is slower
init_scale = 0.2  # weights and biases are initialized normally distributed, scale of gaussian

s_i_clamp, s_ij_clamp = clamped_statistics(data)
s_i_clamp = np.expand_dims(s_i_clamp, axis=1)

for i in range(n_experiments):
    print(f"\nRun: {i}")
    weights, theta, lls = fit_bm_mean_field(allstates, s_i_clamp, s_ij_clamp, lr, schedule, init_scale, compare_explicit=False)
    exp_lls.append(lls)


plt.figure()
for l in exp_lls:
    plt.plot(range(len(l)), l)
plt.xlabel("Iteration")
plt.ylabel("data log likelihood")
# plt.savefig("bm_mean_field_synthetic_convergence.png")

