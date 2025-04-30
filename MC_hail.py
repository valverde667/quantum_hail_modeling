import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import nbinom
import seaborn as sns
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
n_years = 10000  # Number of years to simulate
n_weeks = 48  # Number of weeks in a year
np.random.seed(42)
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
simulated_damage = []
for i in range(n_years):
    # Sample number of hail events for the year
    n_events = nbinom.rvs(r, p)

    if n_events == 0:
        simulated_damage.append(0)
        continue

    # Sample the weeks an event occurs and the size of the hail
    sampled_weeks = np.random.choice(all_weeks, n_events, p=event_pmf)
    sampled_sizes = np.random.choice(hail_size, n_events, p=hail_pmf)

    # Look up the damage for the sampled sizes
    damage = damage_lookup(sampled_sizes)

    # Calculate damage for the year
    simulated_damage.append(damage.sum())

simulated_damage = np.array(simulated_damage)

# ------------------------------------------------------------------------------
#    Results
# Calculate various metrics for the simulated data.
# ------------------------------------------------------------------------------
loss = simulated_damage.mean()
var_95 = np.percentile(simulated_damage, 95)
tvar_95 = simulated_damage[simulated_damage > var_95].mean()
