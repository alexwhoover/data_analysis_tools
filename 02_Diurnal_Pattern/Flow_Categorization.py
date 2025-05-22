# %%

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# rolling_window_hr: Defines the rolling sum window size (storm duration) in hours to calculate the rainfall intensity.
# Closely related to min_intensity_mm_hr, as a larger storm duration will lead to lower rainfall intensities.
rolling_window_hr = 6 # hrs

# min_intensity_mm_hr: Defines the minimum rainfall intensity threshold in mm/hr which defines a rainfall event.
min_intensity_mm_hr = 0.1 # mm/hr

# inter_event_duration_hr: Defines time in hours between storm events in which storm events should be considered as one.
# This parameter is assessed before lead_time_hr and response_time_hr are applied to the storm event start and end times.
inter_event_duration_hr = 24 # hrs

# min_event_duration_hr: Defines minimum storm event duration.
# This parameter is assessed before lead_time_hr and response_time_hr are applied to the storm event start and end times.
min_event_duration_hr = 1 # hrs

# response_time_hr: Defines time in hours added to the end of a storm event to account for runoff response being delayed.
# This parameter should be large enough to capture the time between the last drop of rainfall in a storm and the last drop of flow in sewer as a result of storm event.
response_time_hr = 72 # hrs

# lead_time_hr: Defines time in hours subtracted from the start of a storm event.
lead_time_hr = 2
# %%
# Define Functions
"""
Function to calculate flow events based on rainfall

Inputs:
- df_input (pd.DataFrame): input dataframe with columns 'timestamp', 'rainfall_mm', 'flow_lps'
- rolling_window_hr (float): Defines the rolling sum window size (storm duration) in hours to calculate the rainfall intensity.
- min_intensity_mm_hr (float): Defines the minimum rainfall intensity threshold in mm/hr which defines a rainfall event.
- inter_event_duration_hr (float): Defines time in hours between storm events in which storm events should be considered as one.
- min_event_duration_hr (float): Defines minimum storm event duration.
- response_time_hr (float): Defines time in hours added to the end of a storm event to account for runoff response being delayed.
- lead_time_hr (float): Defines time in hours subtracted from the start of a storm event.

Output:
- df_input (pd.DataFrame) with additional columns rainfall_roll_sum, rainfall_roll_sum_intensity, wet_weather_event, missing_data
- wet_weather_event is either 0 or 1 and defines if the current timestamp is within a storm event
"""

def calculate_flow_events(df_input, rolling_window_hr = 6, min_intensity_mm_hr = 0.1, inter_event_duration_hr = 24, min_event_duration_hr = 1, response_time_hr = 24, lead_time_hr = 2):
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
df_events = calculate_flow_events(df_input, min_intensity_mm_hr = min_intensity_mm_hr, rolling_window_hr = rolling_window_hr, response_time_hr = response_time_hr, inter_event_duration_hr = inter_event_duration_hr, min_event_duration_hr = min_event_duration_hr, lead_time_hr = lead_time_hr)

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
df_events.to_csv("Output_Data/flow_categorization.csv", index = False)
pio.write_html(fig, "Output_Data/flow_categorization_plot.html")
# %%
