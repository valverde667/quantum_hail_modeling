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

# Save the dataframe to a CSV file
df.to_csv("concat_hail_data.csv", index=False)
