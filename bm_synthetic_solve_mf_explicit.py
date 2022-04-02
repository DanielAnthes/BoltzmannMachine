import matplotlib.pyplot as plt
import numpy as np
from bm import fit_bm_mean_field_direct, clamped_statistics


eps = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
print(f"epsilons: {eps}")
lls = list()

# generate all states
nneurons = 10  
allstates = [str(bin(x))[2:].zfill(nneurons) for x in list(range(2**nneurons))]
allstates = [[1 if x == '1' else -1 for x in y] for y in allstates]
allstates = np.array(allstates, dtype=np.float64)

# generate dataset
ndata = 500
idx = np.random.choice(np.arange(2**10), replace=True, size=ndata)
data = allstates[idx]

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

