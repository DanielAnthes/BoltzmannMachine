import matplotlib.pyplot as plt
import numpy as np
from bm import fit_bm_stochastic, clamped_statistics

# EXPERIMENT PARAMETERS
experiments = [10, 100, 1000, 2000, 5000, 10000]
# experiments = [1000]
exp_results = list()
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

### HYPERPARAMETERS
lr = 0.001  # initial learning rate
schedule = 1000  # schedule, larger is slower
init_scale = 0.5  # weights and biases are initialized normally distributed, scale of gaussian
###

s_i_clamp, s_ij_clamp = clamped_statistics(data)
s_i_clamp = np.expand_dims(s_i_clamp, axis=1)


for i, n_glauber in enumerate(experiments):
    print(f"\nRun: {i}")
    weights, theta, lls, weight_diffs = fit_bm_stochastic(allstates, s_i_clamp, s_ij_clamp, lr, n_glauber, schedule, init_scale, compare_explicit=True)
    exp_lls.append(lls)
    exp_results.append(weight_diffs)

plt.figure()
n_exp = len(exp_results)
plt.subplot(1,2,1)
for i in range(n_exp):
    data = exp_lls[i]
    plt.plot(range(len(data)), data)
plt.title("log likelihoods")
plt.ylabel("log_likelihood")
plt.xlabel("iteration")
plt.legend([str(x) for x in experiments])

plt.subplot(2,2,2)
for i in range(n_exp):
    data = exp_results[i]
    plt.plot(range(len(data)), data)
plt.title("max. abs. weight grad. diff.")
plt.xlabel("iteration")
plt.ylabel("max. abs. diff")
plt.savefig("grad_accuracy_synthetic_data_worse_init_no_burnin.png")
plt.show()



