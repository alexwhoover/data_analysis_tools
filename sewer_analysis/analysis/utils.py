import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import holidays

def categorize_flow(df_input, rolling_window_hr, min_intensity_mm_hr, inter_event_duration_hr, min_event_duration_hr, response_time_hr, lead_time_hr):
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


def plot_categorization(df_input):
    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_input["timestamp"],
        y=df_input["flow_lps"],
        mode="lines",
        name="Flow (L/s)",
        line=dict(color="blue", width=1)
    ))

    # Add shaded regions for wet weather events
    wet_weather_events = df_input[df_input["wet_weather_event"] == 1]
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
    missing_data_periods = df_input[df_input["missing_data"] == 1]
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


"""
Purpose: Calculate base flow from a diurnal pattern using the Stevens - Schutzback Method
Inputs:
    MDF (float): Minimum Daily Flow Rate (L/s) (I am calling this MNF)
    ADF (float): Average Daily Flow Rate (L/s) (I am calling this ADWF)

Returns:
    BI (float): Base Infiltration (L/s) (I am calling this GWI)

Source: 
    Mitchell, Paul & Stevens, Patrick & Nazaroff, Adam. (2007). QUANTIFYING BASE INFILTRATION IN SEWERS: A Comparison of Methods and a Simple Empirical Solution. Proceedings of the Water Environment Federation. 2007. 219-238. 10.2175/193864707787974805. 

"""
def calc_base_flow(MDF: float, ADF: float):
    conversion_factor = 0.022824465227271 # 0.023mgd = 1L/s

    # Convert lps to mgd
    MDF_mgd = MDF * conversion_factor
    ADF_mgd = ADF * conversion_factor

    # Stevens-Shutzbach Method Equation
    BI_mgd = (0.4 * MDF_mgd) / (1 - 0.6 * ((MDF_mgd / ADF_mgd)**(ADF_mgd**0.7)))
    
    # Convert mgd to lps
    BI = BI_mgd / conversion_factor

    return BI

def calculate_diurnal(df_input: pd.DataFrame, separate_fridays: bool):
    # Create columns for date, time of day, and weekday based on the timestamp column
    # These columns are used for later processing of diurnal pattern
    df_input['date'] = df_input['timestamp'].dt.date
    df_input['time_of_day'] = df_input['timestamp'].dt.strftime('%H:%M')
    df_input['weekday'] = df_input['timestamp'].dt.strftime('%A')

    # Identify fully dry days
    # A day is considered dry if all 288 5-minute timesteps in a day were previously classified as dry and there is no missing data.
    full_dry_days_series = df_input.groupby(df_input['date']).apply(
        lambda day: (
            day['wet_weather_event'].eq(0).all() and 
            day['missing_data'].eq(0).all() and
            len(day) == 288 # Ensure exactly 288 timesteps in the day (# of 5-min time intervals in 24hrs)
            ),
            include_groups = False
    ).rename('fully_dry') # Returns pd.Series indexed by date

    full_dry_days = full_dry_days_series[full_dry_days_series].index.tolist()

    # Filter input dataframe to only include fully dry days
    # This dataframe will form the basis of diurnal pattern calculation
    df_dwf = df_input[df_input['timestamp'].dt.date.isin(full_dry_days)]

    # Calculate DWF Diurnal Pattern based on selected dry days from previous step

    # Filter flow data based on selected days
    # TODO Add in date exclusion
    #df_dwf_filtered = df_dwf[df_dwf["date"].isin(pd.to_datetime(selected_days).dt.date)].copy()
    df_dwf_filtered = df_dwf.copy()

    # Get BC holidays for the relevant years
    years = df_dwf_filtered["timestamp"].dt.year.unique()
    bc_holidays = set(holidays.CA(subdiv = "BC", years = years).keys())

    if separate_fridays == True:
        df_dwf_filtered.loc[:, "group"] = df_dwf_filtered["date"].apply(
            lambda x: "Weekend/Holiday" 
            if (x in bc_holidays or x.weekday() >= 5) 
            else "Friday" if x.weekday() == 4  
            else "Workday"
        )
    else:
        df_dwf_filtered.loc[:, "group"] = df_dwf_filtered["date"].apply(
            lambda x: "Weekend/Holiday" 
            if (x in bc_holidays or x.weekday() >= 5) 
            else "Workday")

    # Group and calculate mean
    df_diurnal = df_dwf_filtered.groupby(["group", "time_of_day"], as_index = False)["flow_lps"].mean()
    df_diurnal.rename(columns = {"flow_lps": "DWF"}, inplace = True)
    df_diurnal["MNF"] = df_diurnal.groupby("group")["DWF"].transform("min") # Minimum Nighttime Flow (MNF)
    df_diurnal["ADWF"] = df_diurnal.groupby("group")["DWF"].transform("mean") # Average Dry Weather Flow (ADWF)
    df_diurnal["GWI"] = df_diurnal.apply(lambda row: calc_base_flow(row["MNF"], row["ADWF"]), axis = 1) # Ground Water Infiltration
    df_diurnal["SF"] = df_diurnal["DWF"] - df_diurnal["GWI"]

    return df_diurnal, df_dwf_filtered

def plot_diurnal(df_diurnal, df_dwf_filtered):
    # Get unique groups
    groups = df_diurnal["group"].unique()

    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(groups), figsize=(16, 9), sharey=True)

    # Ensure axes is iterable even if there's only one group
    if len(groups) == 1:
        axes = [axes]

    # Loop through each group and plot
    for ax, group in zip(axes, groups):
        subset_diurnal = df_diurnal[df_diurnal["group"] == group]
        subset_data = df_dwf_filtered[df_dwf_filtered["group"] == group]

        for date, subgroup in subset_data.groupby("date"):
            ax.plot(subgroup["time_of_day"], subgroup["flow_lps"], color = "grey", alpha = 0.5)
        
        line_dwf, = ax.plot(subset_diurnal["time_of_day"], subset_diurnal["DWF"], label="Flow", color="blue")
        line_gwi, = ax.plot(subset_diurnal["time_of_day"], subset_diurnal["GWI"], linestyle="--", color="red", label="BI")
        

        ax.set_title(group)
        ax.set_xlabel("Time of Day")
        ax.set_xticks(subset_diurnal["time_of_day"].unique()[::24])
        ax.tick_params(axis="x", rotation=45)
        ax.grid()
        ax.margins(x = 0)
        ax.legend([line_dwf, line_gwi], ['DWF Pattern', 'GWI'], loc="upper right")


    # Set common labels
    fig.supylabel("Flow (L/s)")
    fig.suptitle("Diurnal Flow Patterns", fontsize=18)

    plt.tight_layout()
    plt.show()