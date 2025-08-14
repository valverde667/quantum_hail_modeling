# This script is dedicated to performing the MC simulations with a calibrated
# gamma curve. Becuase of the complexity, it was thought best to keep this
# implementation separate from the MC_hail.py script for clarity. The MC_hail
# script will need to be run in order to get the annual variance in loss from the
# lookup table, as well as the assigned parameter values for the negative binomial.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import nbinom
import os
import datetime
import glob

from library import set_plot_style

# Set plot style
set_plot_style()

# ------------------------------------------------------------------------------
#    Variables
# Define any useful variables here.
# ------------------------------------------------------------------------------
n_years = 100000  # Number of years to simulate
n_weeks = 48  # Number of weeks in a year
SEED = 42
rng = np.random.default_rng(SEED)
start_year = 1980
end_year = 2024
time_span = end_year - start_year  # Time span in years

# === Severity model selection & Gamma calibration ===
# Toggle calibration on/off:
USE_GAMMA = False  # if False, use the step lookup severity
CALIBRATE_GAMMA = (
    True  # if True and USE_GAMMA, fit k and θ_i to match lookup means & variance
)
SIZE_THRESHOLD = 2.0  # zero loss for sizes < threshold (mirrors lookup)

# Values from MC_hail script
var_L = 17.29
mu_L = 2.76


# ------------------------------------------------------------------------------
#    Functions
# Section of useful functions
# ------------------------------------------------------------------------------
def damage_function(size):
    # Damage function depending on hail size. Size is assumed to be the diameter
    # of the hail in inches.
    if size < 2:
        return 0
    elif size < 3:
        return 0.1
    elif size < 5:
        return 0.4
    else:
        return 0.9


def calc_neg_binomial_params(data):
    """Calculate negative binomial parameters using method of moments."""
    mu = data.mean()
    var = data.var()  # Fix: use actual sample variance

    if var <= mu:
        # fallback to Poisson assumption
        return np.inf, mu / (mu + 1e-9)  # p close to 1

    r = mu**2 / (var - mu)
    p = r / (r + mu)
    return r, p


def calibrate_gamma_constant_shape(
    sizes, probs, lookup_fn, var_target, size_threshold=None, eps=1e-9
):
    """
    Calibrate a constant-shape Gamma mixture so that:
      - For each size s_i: E[X|S=s_i] matches the lookup mean mu_i.
      - Unconditional per-event variance equals var_target.
    Returns: (k, theta_map) with k>0 and theta_map[size]=theta_i.
    """
    sizes = np.asarray(sizes, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("hail-size PMF must sum to 1.")
    # Conditional means from lookup (zero below threshold if requested)
    mu_i = lookup_fn(sizes).astype(float)
    if size_threshold is not None:
        mu_i = np.where(sizes < size_threshold, 0.0, mu_i)
    # Mixture moments of mu(S)
    mu_bar = np.sum(probs * mu_i)
    Emu2 = np.sum(probs * (mu_i**2))
    var_mu = Emu2 - mu_bar**2
    # Feasibility
    if not (var_target > var_mu + eps):
        raise ValueError(
            f"Target per-event variance too small: var_target={var_target:.6g} "
            f"<= Var(mu(S))={var_mu:.6g}. Increase var_target or revise mu_i."
        )
    # Solve for constant shape k from law of total variance
    k = Emu2 / (var_target - var_mu)
    # Per-size scales ensuring E[X|s_i]=mu_i
    theta_i = np.where(mu_i > 0, mu_i / k, 0.0)
    theta_map = dict(zip(sizes.tolist(), theta_i.tolist()))
    return float(k), theta_map


def make_gamma_sampler_calibrated(k, theta_map, rng):
    """
    Vectorized sampler for constant-shape (k) Gamma with per-size scales theta_i.
    Sizes not exactly in theta_map use nearest neighbor mapping.
    """
    size_keys = np.array(list(theta_map.keys()), dtype=float)
    theta_vals = np.array([theta_map[s] for s in size_keys], dtype=float)

    # Small helper to fetch theta for arbitrary sizes
    def theta_for(arr):
        arr = np.asarray(arr, dtype=float)
        out = np.empty_like(arr, dtype=float)
        # exact matches
        lookup = {sk: tv for sk, tv in zip(size_keys.tolist(), theta_vals.tolist())}
        mask = np.isin(arr, size_keys)
        if mask.any():
            out[mask] = np.array([lookup[v] for v in arr[mask]], dtype=float)
        if (~mask).any():
            idx = np.abs(arr[~mask, None] - size_keys[None, :]).argmin(axis=1)
            out[~mask] = theta_vals[idx]
        return out

    def sampler(sizes):
        s = np.asarray(sizes, dtype=float)
        theta = theta_for(s)
        out = np.zeros_like(s, dtype=float)
        mask = theta > 0
        if mask.any():
            out[mask] = rng.gamma(shape=k, scale=theta[mask], size=mask.sum())
        return out

    return sampler


# ------------------------------------------------------------------------------
#    Ingest and Process Data
# Ingest data from concatenated csv file (see hail_data_processing.py) and
# perform any operations like converting to datetime, etc.
# ------------------------------------------------------------------------------
df = pd.read_csv("concat_hail_data.csv")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Convert the BEGIN_DATE and END_DATE columns to datetime objects
df["BEGIN_DATE"] = pd.to_datetime(df["BEGIN_DATE"])
df["END_DATE"] = pd.to_datetime(df["END_DATE"])

# Create a column of only the year and find hail events per year
df["YEAR"] = df["BEGIN_DATE"].dt.year
events_per_year = df.groupby("YEAR").size()

# Calculate the negative binomial parameters
r, p = calc_neg_binomial_params(events_per_year)

# ------------------------------------------------------------------------------
#    Probability Mass Function (PMF)
# Create PMFs for hail event and hail size using the historical data
# ------------------------------------------------------------------------------
# Collect the weekly counts over all years
weekly_counts = df.groupby("MONTH_WEEK_INDEX").size()
all_weeks = np.arange(1, 49)
weekly_counts = weekly_counts.reindex(all_weeks, fill_value=0)
avg_weekly_counts = weekly_counts / time_span
event_pmf = weekly_counts / weekly_counts.sum()

# Create size pmf
size_counts = df.groupby("MAGNITUDE").size().sort_index()
hail_pmf = size_counts / size_counts.sum()
hail_size = size_counts.index.to_numpy()

# Vectorize the damage function
damage_lookup = np.vectorize(damage_function)

if not USE_GAMMA:

    def damage_sampler(arr):
        return damage_lookup(np.asarray(arr, dtype=float))

else:
    if CALIBRATE_GAMMA:
        # Count model moments from NegBin (already computed as r, p)
        EN = r * (1 - p) / p
        VarN = r * (1 - p) / (p**2)

        # Per-event mean from lookup
        mu_i = damage_lookup(hail_size).astype(float)
        mu_i = np.where(hail_size < SIZE_THRESHOLD, 0.0, mu_i)
        mu_bar = np.sum(hail_pmf * mu_i)

        # Back out per-event variance target via compound identity
        varX_target = (var_L - VarN * (mu_bar**2)) / EN

        # Diagnostics
        Emu2 = np.sum(hail_pmf * (mu_i**2))
        var_mu = Emu2 - mu_bar**2
        print(f"[Gamma calibration] EN={EN:.6g}, VarN={VarN:.6g}")
        print(f"[Gamma calibration] mu_bar (per-event mean from lookup) = {mu_bar:.6g}")
        print(f"[Gamma calibration] Var(mu(S)) = {var_mu:.6g}")
        print(
            f"[Gamma calibration] Var(L)_target = {var_L:.6g}  ->  Var(X)_target = {varX_target:.6g}"
        )

        # Calibrate k and θ_i
        k, theta_map = calibrate_gamma_constant_shape(
            sizes=hail_size,
            probs=hail_pmf,
            lookup_fn=damage_lookup,
            var_target=varX_target,
            size_threshold=SIZE_THRESHOLD,
        )
        print(f"[Gamma calibration] Solved constant shape k = {k:.6g}")

        damage_sampler = make_gamma_sampler_calibrated(k, theta_map, rng)

    else:
        # Uncalibrated Gamma: constant k and exponential θ(s) as a fallback
        k = 3.0
        c = 100.0
        gamma_scale = 0.8

        def damage_sampler(arr):
            s = np.asarray(arr, dtype=float)
            theta = c * np.exp(gamma_scale * s)
            out = np.zeros_like(s, dtype=float)
            mask = s >= SIZE_THRESHOLD
            out[mask] = rng.gamma(shape=k, scale=theta[mask], size=mask.sum())
            return out


stop
# ------------------------------------------------------------------------------
#    MC Simulation
# Simulate hail events for a year using the PMFs to sample events and sizes.
# Each MC run is for a year totaling 48 weeks. For each year, the number of hail
# events is sampled from a negative binomial distribution. Given the number of
# events, the PMF for weekly events and hail size is used to distribute the
# events per year and hail size.
# ------------------------------------------------------------------------------
losses = []
for i in range(n_years):
    # Sample number of hail events for the year
    n_events = rng.negative_binomial(r, p)

    if n_events == 0:
        losses.append(0)
        continue

    # Sample the weeks an event occurs and the size of the hail
    sampled_weeks = np.random.choice(all_weeks, n_events, p=event_pmf)
    sampled_sizes = np.random.choice(hail_size, n_events, p=hail_pmf)

    # # Look up the damage for the sampled sizes
    # # to be filled with calibrated gamma
    # damage = damage_sampler(sampled_sizes)

    # # Calculate damage for the year
    # losses.append(damage.sum())

losses = np.array(losses)

# ------------------------------------------------------------------------------
#    Results
# Calculate various metrics for the simulated data.
# ------------------------------------------------------------------------------
loss = losses.mean()
var_95 = np.percentile(losses, 95)
tvar_95 = losses[losses > var_95].mean()

print(f"Mean Loss: {loss:.2f}")
print(f"95th Percentile Loss: {var_95:.2f}")
print(f"Tail 95th Percentile Loss: {tvar_95:.2f}")
print(f"Max Loss: {losses.max():.2f}")


# Plot cumulative distribution function (CDF) of losses
sorted_losses = np.sort(losses)
cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

plt.plot(sorted_losses, cdf)
plt.xlabel("Loss")
plt.ylabel("Cumulative Probability")
plt.tight_layout()
plt.savefig("losses_cdf.pdf", dpi=800, bbox_inches="tight")
plt.show()

# Plot histogram of losses
counts, bin_edges = np.histogram(losses, bins=50, density=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], edgecolor="black", lw=1)
plt.xlabel("Loss (arb.)")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.savefig("losses_histogram.pdf", dpi=800, bbox_inches="tight")
plt.show()

# Slice off the first bin
counts = counts[1:]
bin_centers = bin_centers[1:]
bin_width = bin_edges[1] - bin_edges[0]

# Plot
plt.bar(bin_centers, counts, width=bin_width, edgecolor="black", lw=1)
plt.tight_layout()
plt.savefig("losses_histogram_no_zero.pdf", dpi=800, bbox_inches="tight")
plt.show()
