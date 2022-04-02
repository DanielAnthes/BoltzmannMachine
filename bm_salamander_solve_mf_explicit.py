import matplotlib.pyplot as plt
import numpy as np
from bm import fit_bm_mean_field_direct, clamped_statistics

eps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
print(f"epsilons: {eps}")
lls = list()

nneurons = 160
data = np.loadtxt("bint.txt")
data = data[:nneurons,:953]  # select first n neurons, select only one trial
data[data<.5] = -1  # convert to -1,1 format
data = data.T  # transpose to fit erxpected data shape
data = np.ascontiguousarray(data)  # contiguous in memory for performance


s_i_clamp, s_ij_clamp = clamped_statistics(data)
s_i_clamp = np.expand_dims(s_i_clamp, axis=1)

for e in eps:
    weights, theta, likelihood = fit_bm_mean_field_direct(data, s_i_clamp, s_ij_clamp, epsilon=e)
    print(likelihood)
    lls.append(likelihood)

plt.figure()
plt.plot(eps, lls)
plt.xscale('log')
plt.xlabel("epsilon")
plt.ylabel("log likelihood")
plt.title("explicit solution for BM using mean field equation")
plt.savefig("explicit_solutions_eps.png")
plt.show()
