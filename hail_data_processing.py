# The following script will ingest the downloaded hail data and process it. The
# data was collected from https://www.ncdc.noaa.gov/stormevents/ focusing on
# Tarrant county which is where the Fort Worth area is. Since the data can only
# be viewed 500 rows at a time, multiple date ranges were used to collect the
# data. We will combine the data from 1980 to the end of 2024 into a single
# dataframe.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime
import glob

# Plot style settings

# Set the style of seaborn
sns.set_style("whitegrid")

# Set the color palette
sns.set_palette("pastel")

# Set the font size
plt.rcParams["font.size"] = 14

# Set font to Helvetica Neue
plt.rcParams["font.family"] = "Helvetica Neue"

# Set font weight to light
plt.rcParams["font.weight"] = "light"


# ------------------------------------------------------------------------------
#    Ingest Data and Cleaning
# Extract data from multiple CSV files and combine them into a single dataframe.
# ------------------------------------------------------------------------------
# Set the path to the directory where the CSV files are located
data_path = os.path.join(os.getcwd(), "data")

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(data_path, "hail_data*"))

# Initialize an empty list to store the dataframes
dfs = []

# List of columns to keep
columns_to_keep = [
    "BEGIN_DATE",
    "END_DATE",
    "BEGIN_LAT",
    "BEGIN_LON",
    "END_LAT",
    "END_LON",
    "MAGNITUDE",
]

# Loop through each CSV file and read it into a dataframe
for file in csv_files:
    df = pd.read_csv(file)

    # Drop unecessary columns
    df = df[columns_to_keep]

    # Convert the BEGIN_DATE and END_DATE columns to datetime objects
    df["BEGIN_DATE"] = pd.to_datetime(df["BEGIN_DATE"])
    df["END_DATE"] = pd.to_datetime(df["END_DATE"])

    dfs.append(df)

# Concatenate all the dataframes into a single dataframe
df = pd.concat(dfs, ignore_index=True)

# Remove any rows that do not have a magntitude, Begin or End date
df = df.dropna(subset=["MAGNITUDE", "BEGIN_DATE", "END_DATE"])

# Convert the MAGNITUDE column to numeric values
df["MAGNITUDE"] = pd.to_numeric(df["MAGNITUDE"], errors="coerce")

# Remove any rows that have a magnitude of 0
df = df[df["MAGNITUDE"] != 0]


# Save the dataframe to a CSV file
df.to_csv("concat_hail_data.csv", index=False)


# ------------------------------------------------------------------------------
#    Visualize Data
# Visualize various columns in dataframe.
# ------------------------------------------------------------------------------
# Set the figure size
plt.figure(figsize=(10, 6))

# Plot a histogram of the magnitude column. Since magnitudes come in fixed values
# bin based on the unique values.
size_freq = df["MAGNITUDE"].value_counts().sort_index()
plt.scatter(size_freq.index, size_freq.values)
plt.xlabel("Hail Size (cm)")
plt.ylabel("Frequency")
plt.savefig("hail_size_histogram.pdf", dpi=500)
plt.show()

# Aggregate the data by week over all years from Jan to Dec (calendar weeks)
df["MONTH"] = df["BEGIN_DATE"].dt.month
df["WEEK_IN_MONTH"] = df["BEGIN_DATE"].dt.day.apply(lambda x: (x - 1) // 7 + 1)
df["MONTH_WEEK_INDEX"] = (df["MONTH"] - 1) * 4 + df["WEEK_IN_MONTH"]

# Create a series with index of week numbers (1-48) and values as the count of 
# hail events in each week
weekly_counts = df.groupby("MONTH_WEEK_INDEX").size()

# Initialize a series with all weeks (1-48) and set the counts to 0 then fill
all_bins = pd.Series(0, index=np.arange(1, 49))
weekly_counts = all_bins.add(weekly_counts, fill_value=0)

# Create histogram-style bar plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(weekly_counts.index, weekly_counts.values, width=0.8, edgecolor="black")
ax.set_xlabel("Week Number")
ax.set_ylabel("Number of Hail Events")

# Major ticks (every week)
ax.set_xticks(np.arange(1, 49))  # 1 through 48

# Minor ticks with month labels every 4 weeks
month_labels = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
month_ticks = [i * 4 + 1 for i in range(12)]
plt.xticks(month_ticks, month_labels)

plt.tight_layout()
plt.savefig("hail_events_per_week.pdf", dpi=500)
plt.show()
