import marimo as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set number of trials
n_samples = 1000000

# Choose the distribution to sample from
distribution = "chi_squared"  # Options: "normal", "uniform", "student_t", "custom"

# Sampling logic
if distribution == "normal":
    data = np.random.normal(loc=0, scale=1, size=n_samples)

elif distribution == "uniform":
    data = np.random.uniform(low=-3, high=3, size=n_samples)

elif distribution == "chi_squared":
    df = 4  # degrees of freedom
    data = np.random.chisquare(df=df, size=n_samples)

elif distribution == "custom":
    # Bimodal base
    data1 = np.random.normal(loc=-2, scale=0.5, size=n_samples // 2)
    data2 = np.random.poisson(lam=5, size=n_samples // 2)
    # Add random noise (shift per sample)
    noise = np.random.normal(loc=0, scale=0.72, size=n_samples)
    data = np.concatenate([data1, data2]) + noise

else:
    raise ValueError(f"Unknown distribution type: {distribution}")

# Plot the histogram and KDE
plt.figure(figsize=(6, 4))
sns.histplot(data, bins=50, kde=False, stat="density", color="steelblue")
# plt.title(f"Sampling from {distribution.capitalize()} Distribution ({n_samples} samples)")
# plt.xlabel("Value")
# plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.savefig("distribution.pdf")
plt.show()
