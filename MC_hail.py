import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import nbinom
import os
import datetime
import glob
from dataclasses import dataclass

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


@dataclass
class GammaDamageConfig:
    base_scale: float = 100.0
    shape_intercept: float = 2.0
    shape_slope: float = 0.3
    scale_growth: float = 0.8  # exponent coefficient for scale
    round_to: int | None = 2  # Set None for rounding.


def build_damage_sampler(
    model: str = "lookup",
    *,
    gamma_cfg: GammaDamageConfig = GammaDamageConfig(),
    rng: np.random.Generator = rng,
):
    """
    Return a function f(sizes: array_like) -> per-event damagese (array).
    Model: "lookup" (step model) or "gamma" (heavy-tail severity).
    """
    if model == "lookup":

        def sampler(sizes):
            s = np.asarray(sizes, dtype=float)
            return damage_lookup(s)

        return sampler

    if model == "gamma":

        def sampler(sizes):
            s = np.asarray(sizes, dtype=float)

            # Handle NaNs gracefully as zero damage
            mask = ~np.isnan(s)
            out = np.zeros_like(s, dtype=float)
            if mask.any():
                k = gamma_cfg.shape_intercept + gamma_cfg.shape_slope * s[mask]
                theta = gamma_cfg.base_scale * np.exp(gamma_cfg.scale_growth * s[mask])
                draws = rng.gamma(shape=k, scale=theta, size=k.shape)
                if gamma_cfg.round_to is not None:
                    draws = np.round(draws, gamma_cfg.round_to)
                out[mask] = np.clip(draws, 0, None)
            return out

        return sampler

    raise ValueError("model must be 'lookup' or 'gamma'")


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
size_counts = df.groupby("MAGNITUDE").size()
hail_pmf = size_counts / size_counts.sum()
hail_size = df["MAGNITUDE"].unique()

# Vectorize the damage function
damage_lookup = np.vectorize(damage_function)

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
    n_events = nbinom.rvs(r, p)

    if n_events == 0:
        losses.append(0)
        continue

    # Sample the weeks an event occurs and the size of the hail
    sampled_weeks = np.random.choice(all_weeks, n_events, p=event_pmf)
    sampled_sizes = np.random.choice(hail_size, n_events, p=hail_pmf)

    # Look up the damage for the sampled sizes
    damage = damage_lookup(sampled_sizes)

    # Calculate damage for the year
    losses.append(damage.sum())

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
