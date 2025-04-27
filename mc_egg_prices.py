# Re-import necessary libraries after code state reset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import random
import os

# ------------------------------------------------------------------------------
#    Useful Variables and/or Switches
# ------------------------------------------------------------------------------
use_analytic = False  # Use analytic distribution for MC simulations
use_historical = True  # Use historical data to estimate parameters

# ------------------------------------------------------------------------------
#    Simulate Egg Prices Using Ornstein-Uhlenbeck Process
# The Ornstein-Uhlenbeck process is a stochastic process that describes the
# evolution of a variable that tends to revert to a long-term mean over time.
# It is often used in finance to model mean-reverting processes, such as interest
# rates or commodity prices. In this case, we will use it to simulate the price
# of eggs over a year.
# ------------------------------------------------------------------------------
# Parameters for Ornstein-Uhlenbeck process
if use_analytic:
    np.random.seed(42)
    n_days = 365
    n_simulations = 10000
    mu = 3.5  # long-term mean price ($)
    theta = 0.01  # rate of mean reversion
    sigma = 0.2  # volatility
    dt = 1  # 1 day time step
    initial_price = 2.5  # starting price

    # Simulate OU process
    ou_price_paths = np.zeros((n_simulations, n_days))
    for i in range(n_simulations):
        prices = [initial_price]
        for _ in range(1, n_days):
            prev = prices[-1]
            dP = theta * (mu - prev) * dt + sigma * np.random.normal(scale=np.sqrt(dt))
            prices.append(prev + dP)
        ou_price_paths[i, :] = prices

    # Plot the OU-based price paths
    if n_simulations < 20:
        fig, ax = plt.subplots(figsize=(10, 6))
        for path in ou_price_paths:
            ax.plot(path, alpha=0.8)
        ax.set_title("Egg Price Simulation Using Ornstein–Uhlenbeck Process")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price ($)")
        plt.tight_layout()
        plt.show()

    # Option 1: Mean and Std Dev per Day
    daily_means = np.mean(ou_price_paths, axis=0)
    daily_stds = np.std(ou_price_paths, axis=0)

    # Option 1b: Mean and Std Dev per trial
    trial_means = np.mean(ou_price_paths, axis=1)
    trial_stds = np.std(ou_price_paths, axis=1)

    # Option 2: Total Yearly Cost Per Simulation
    total_costs = np.sum(ou_price_paths, axis=1)
    mean_total_cost = np.mean(total_costs)
    std_total_cost = np.std(total_costs)

    # Plot daily mean with shaded uncertainty
    fig, ax = plt.subplots(figsize=(10, 6))
    days = np.arange(1, n_days + 1)
    ax.plot(days, daily_means, label="Mean Daily Price", color="blue")
    ax.fill_between(
        days,
        daily_means - daily_stds,
        daily_means + daily_stds,
        color="blue",
        alpha=0.3,
        label="±1 Std Dev",
    )
    ax.set_title("Daily Egg Price: Mean and Uncertainty")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("daily_price_mean_std.pdf", dpi=500)
    plt.show()

    # Plot histogram of total yearly cost
    fig, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(total_costs, bins=30, color="green", edgecolor="black", alpha=0.7)
    ax2.axvline(
        mean_total_cost,
        color="red",
        linestyle="--",
        label=f"Mean = ${mean_total_cost:.2f}",
    )
    ax2.set_title("Histogram of Total Yearly Egg Cost")
    ax2.set_xlabel("Total Cost ($)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("total_cost_histogram.pdf", dpi=500)
    plt.show()

# ------------------------------------------------------------------------------
#    Use Historical Data to Estimate Parameters
# Using data downloaded from https://fred.stlouisfed.org/series/APU0000708111,
# we can perform MC simulations by sampling from a PDF that is created from 25
# years of egg prices per day.
# ------------------------------------------------------------------------------
if use_historical:
    # Load historical egg price data and adjust for inflation
    # Merge egg price data with CPI data on DATE
    egg_prices = pd.read_csv("egg_prices.csv", parse_dates=["DATE"])
    cpi_data = pd.read_csv("cpi_data.csv", parse_dates=["DATE"])

    cpi_data = cpi_data.rename(columns={"CPIAUCSL": "CPI"})
    merged_data = pd.merge(egg_prices, cpi_data, on="DATE", how="left")

    # Get CPI for the most recent date to use as the base (assume 2025)
    cpi_base = merged_data["CPI"].iloc[-1]

    # Adjust nominal prices to real (2025) dollars
    merged_data["Real_Price_2025"] = merged_data["Price"] * (
        cpi_base / merged_data["CPI"]
    )

    # Plot the adjusted value and the nominal value.
    plt.figure(figsize=(10, 6))
    plt.plot(
        merged_data["DATE"], merged_data["Price"], label="Nominal Price", alpha=0.7
    )
    plt.plot(
        merged_data["DATE"],
        merged_data["Real_Price_2025"],
        label="Real Price (2025 Dollars)",
        alpha=0.7,
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price per Dozen Eggs ($)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("adjusted_price.pdf", dpi=500)
    plt.show()

    # Create a probability density function (PDF) from the historical data
    # Use the adjusted prices for the PDF
    # Extract monthly inflation-adjusted egg prices
    monthly_prices = merged_data["Real_Price_2025"].dropna()
    monthly_prices = monthly_prices[monthly_prices > 0]  # remove non-positive values

    # Estimate a probability density function using kernel density estimation
    kde = gaussian_kde(monthly_prices)

    # Create a range of values for plotting the KDE
    x_vals = np.linspace(monthly_prices.min(), monthly_prices.max(), 500)
    pdf_vals = kde(x_vals)

    # Plot the histogram and KDE
    plt.figure(figsize=(10, 6))
    plt.hist(monthly_prices, bins=30, density=True, alpha=0.5, edgecolor="black")
    plt.plot(x_vals, pdf_vals, color="darkblue", lw=2, label="KDE")
    plt.xlabel("Monthly Egg Price ($, adjusted to 2025)", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("egg_price_distribution.pdf", dpi=500)
    plt.show()

    # Use KDE to sample from distribution for 10,000 trials of egg prices.
    n_trials = 10000
    np.random.seed(42)
    sampled_prices = kde.resample(n_trials)[0]
    sampled_prices = np.clip(sampled_prices, 0, None)  # Remove negative prices
    std_sampled = np.std(sampled_prices)
    se_of_std = std_sampled / np.sqrt(2 * (n_trials - 1))

    # Plot histogram of sampled prices
    plt.figure(figsize=(10, 6))
    plt.hist(sampled_prices, bins=30, alpha=0.5, edgecolor="black")
    plt.xlabel("Monthly Egg Price ($, adjusted to 2025)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    plt.savefig("egg_price_distribution_sampled.pdf", dpi=500)
    plt.show()

    print(f"Mean sampled price: ${np.mean(sampled_prices):.2f}")
    print(f"Std sampled price: ${np.std(sampled_prices):.2f}")
    print(f"Standard error of std: ±${se_of_std:.4f}")

    # --------------------------------------------------------------------------
    #   Exponential Simulation
    # --------------------------------------------------------------------------
    # Simulate egg prices using exponential increments over time

# np.random.seed(42)

# n_days = 365
# n_simulations = 10
# base_price = 2.5  # starting price in dollars

# # Exponential distribution parameters
# lambda_exp = 1/0.02  # mean price increase per day = 0.02 dollars

# # Simulate exponential price increases
# price_paths_exp = []
# for _ in range(n_simulations):
#     daily_increases = np.random.exponential(scale=1/lambda_exp, size=n_days)
#     price_series = base_price + np.cumsum(daily_increases)
#     price_paths_exp.append(price_series)

# # Plot the exponential-based price paths
# fig, ax = plt.subplots(figsize=(10, 6))
# for path in price_paths_exp:
#     ax.plot(path, alpha=0.8)
# ax.set_title("Simulated Egg Price Paths Using Exponential Increases")
# ax.set_xlabel("Day")
# ax.set_ylabel("Price ($)")
# plt.tight_layout()@
# plt.show()
