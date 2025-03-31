# %%

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

# %%
# Import data from pre-processing step
fp = 'Input_Data/flow_precip_data.csv'
df_input = (pd.read_csv(fp)
    .assign(timestamp = lambda df: pd.to_datetime(df['timestamp']))
)
# %%
# Define Parameters

# %%
# Define Functions
def calculate_flow_events(df_input, min_intensity_mm_hr = 0.1, rolling_window_hr = 6, response_time_hr = 24, inter_event_duration_hr = 24, min_event_duration_hr = 1, lead_time_hr = 2):
    window_size = int(rolling_window_hr * 60 / 5)

    # Calculate rolling sum of precip, then convert to intensity
    # Duration of intensity calculation = rolling_window_hr
    df_input['rainfall_roll_sum'] = df_input['rainfall_mm'].rolling(window = window_size).sum()
    df_input['rainfall_roll_sum_intensity'] = df_input['rainfall_roll_sum'] / rolling_window_hr

    # Initialize the wet_weather_event column with 0
    df_input['wet_weather_event'] = 0

    # Identify potential events based on intensity threshold
    # event_indices is a list of all indices with precip >= threshold
    event_indices = df_input.index[
        df_input['rainfall_roll_sum_intensity'] >= min_intensity_mm_hr
    ].tolist()

    # If there are no rainfall events, return df_input
    if not event_indices:
        return df_input
    
    # Combine events within the inter-event duration using timestamps
    current_event_start = df_input.loc[event_indices[0], 'timestamp']
    current_event_end = df_input.loc[event_indices[0], 'timestamp']

    for i in range(1, len(event_indices)):
        current_timestamp = df_input.loc[event_indices[i], 'timestamp']

        # If two timestamps above threshold are within inter-event duration, change event end
        if (current_timestamp - current_event_end).total_seconds() <= (inter_event_duration_hr * 3600):
            current_event_end = current_timestamp

        # Once the next timestamp above threshold is outside of inter-event duration, check if event duration meets minimum requirement
        elif (current_event_end - current_event_start).total_seconds() >= (min_event_duration_hr * 3600):
            # If minimum requirement met, mark as storm
            df_input.loc[
                (df_input['timestamp'] >= (current_event_start - timedelta(hours = lead_time_hr))) & (df_input['timestamp'] <= (current_event_end + timedelta(hours = response_time_hr))),
                'wet_weather_event'
            ] = 1

            current_event_start = current_timestamp
            current_event_end = current_timestamp

    # Add a column showing periods with NA values in flow or precip
    df_input['missing_data'] = (~((~df_input['rainfall_mm'].isnull()) & (~df_input['flow_lps'].isnull()))).astype(int)
    
    return df_input

        
# %%
# Categorize flow as dry or wet
df_events = calculate_flow_events(df_input, min_intensity_mm_hr = 0.1, rolling_window_hr = 6, response_time_hr = 72, inter_event_duration_hr = 24, min_event_duration_hr = 1, lead_time_hr = 2)

# %%
# Create figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_events["timestamp"],
    y=df_events["flow_lps"],
    mode="lines",
    name="Flow (L/s)",
    line=dict(color="blue", width=1)
))

# Add shaded regions for wet weather events
wet_weather_events = df_events[df_events["wet_weather_event"] == 1]
if not wet_weather_events.empty:
    start = None
    for i in range(len(wet_weather_events)):
        if start is None:
            start = wet_weather_events.iloc[i]["timestamp"]
        # Check if it's the last row or if there's a gap in time
        if i == len(wet_weather_events) - 1 or (wet_weather_events.iloc[i+1]["timestamp"] - wet_weather_events.iloc[i]["timestamp"]).seconds > 3600:
            fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=start, x1=wet_weather_events.iloc[i]["timestamp"],
                y0=0, y1=1,  # Full height of the plot
                fillcolor="red",
                opacity=0.3,
                layer="below",
                line_width=0
            )
            start = None  # Reset for next storm event

# Add shaded regions for missing data
missing_data_periods = df_events[df_events["missing_data"] == 1]
if not missing_data_periods.empty:
    start = None
    for i in range(len(missing_data_periods)):
        if start is None:
            start = missing_data_periods.iloc[i]["timestamp"]
        # Check if it's the last row or if there's a gap in time
        if (i == len(missing_data_periods) - 1) or (
            (missing_data_periods.iloc[i+1]["timestamp"] - missing_data_periods.iloc[i]["timestamp"]).total_seconds() > 300
        ):
            fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=start, x1=missing_data_periods.iloc[i]["timestamp"],
                y0=0, y1=1,  # Full height of the plot
                fillcolor="grey",
                opacity=0.3,
                layer="below",
                line_width=0
            )
            start = None  # Reset for next missing data period

# Formatting
fig.update_layout(
    title="Flow Time Series with Wet Weather Events (Red) and Missing Data (Gray)<br><sup>Missing data is defined as periods where either precipitation or flow is missing</sup>",
    xaxis_title="Timestamp",
    yaxis_title="Flow (L/s)",
    xaxis=dict(tickangle=45),
    legend=dict(title="Legend"),
    template="plotly_white"
)

# Show the plot
fig.show()
# %%
# Save data to csv
df_events.to_csv("Output_Data/Flow_Categorization.csv", index = False)
pio.write_html(fig, "Output_Data/Flow_Categorization_Plot.html")
# %%
