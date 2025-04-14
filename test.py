import numpy as np

# Step t
t = 200
x0 = 5

# Linear beta schedule: beta_t for t = 0 to 999
def beta_t(t):
    return 0.0001 + (0.02 - 0.0001) * t / 1000

# Compute all betas up to t-1
betas = np.array([beta_t(s) for s in range(0, 101)])  # s=0,...,199
alphas = 1.0 - betas
alpha_bar = np.prod(alphas)

# Compute mean
mean = np.sqrt(alpha_bar) * x0

# Round to 4 decimal places
print("Final mean:", round(mean, 4))