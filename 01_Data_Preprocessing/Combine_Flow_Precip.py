# %%

import pandas as pd
import numpy as np
import os

# %%
# Filepath Declarations and import

precip_fp = "Input_Data/MCCLEERY.csv"
flow_fp = "Input_Data/ANG_COM002.csv"

df_precip_raw = (pd.read_csv(precip_fp, skiprows = 2)
                .rename(columns = {
                    'yyyy/MM/dd hh:mm:ss': 'timestamp',
                    'FINAL Rainfall (mm)': 'rainfall_mm',
                    'QAQC Data Flags': 'flag'
                })
                .assign(timestamp = lambda df: pd.to_datetime(df['timestamp'], format = '%Y/%m/%d %H:%M:%S'))
)
df_flow_raw = (pd.read_csv(flow_fp, skiprows=2)
            .rename(columns={
                'yyyy/MM/dd hh:mm:ss': 'timestamp',
                'Final Depth (mm)': 'depth_mm',
                'Final Flow (l/s)': 'flow_lps',
                'Final Velocity (m/s)': 'velocity_mps'
            })
            .assign(timestamp = lambda df: pd.to_datetime(df['timestamp'], format = '%Y/%m/%d %H:%M:%S'))
)

# %%
# For simplicity, we will drop all rainfall rows that include a flag != 0, then drop the flag column
df_precip_filtered = (df_precip_raw.query('flag == 0') # Filter rows where flag is 0
                .drop(columns = ['flag']) # Drop flag column
)
# %%
# Create a new dataframe with no missing timestamps
# Join both flow and precip to this new dataframe
# This ensures there are no missing timestamps in data

# Get min and max timestamps
min_time = df_flow_raw['timestamp'].min()
max_time = df_flow_raw['timestamp'].max()

# Create a new dataframe with a timestamp column at 5-minute intervals
timestamps = pd.date_range(
    start = min_time,
    end = max_time,
    freq = '5min' # 5 minutes
)

df = pd.DataFrame({'timestamp': timestamps})

# Merge on new df
df = pd.merge(df, df_flow_raw, on = 'timestamp', how = 'left')
df = pd.merge(df, df_precip_filtered, on = 'timestamp', how = 'left')

# %%
# Write new dataframe to csv
df.to_csv('Output_Data/flow_precip_data.csv', index = False)
# %%
