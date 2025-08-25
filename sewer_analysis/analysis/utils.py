import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import holidays

def categorize_flow(df_input, separate_fridays, rolling_window_hr, min_intensity_mm_hr, inter_event_duration_hr, min_event_duration_hr, response_time_hr, lead_time_hr, plot):
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
    df_input['missing_data'] = (~((~df_input['rainfall_mm'].isnull()) & (~df_input['flow_lps'].isnull()) & (df_input['flow_lps'] != 0))).astype(int)

    # Create wet weather mask for df_input (for later use in runoff ratio calculation)
    wet_weather_mask = (df_input['missing_data'] == 0) & (df_input['wet_weather_event'] == 1)
    
    # Plot categorization at this point before further filtering as a QA/QC check
    if plot == True:
        plot_categorization(df_input)

    # Filter dataframe to only include fully dry days
    df_dwf = _filter_dwf(df_input)
    
    # Categorize days into groups, either [Workday, Weekend/Holiday] or [Workday, Friday, Weekend/Holiday]
    df_dwf = _categorize_days(df_dwf, separate_fridays)

    # Get rid of intermediate calculation columns
    df_dwf = df_dwf[["timestamp", "date", "time_of_day", "group", "flow_lps"]]
    
    return df_dwf, wet_weather_mask

def calculate_IQR(s: pd.Series):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return IQR, lower_bound, upper_bound


def _filter_dwf(df):
    df = df.copy()

    _add_time_columns(df)

    # Initial manual "guess" from categorize_flow()
    # Identify fully dry days: no wet weather, no missing data, and exactly 288 timesteps (5-min intervals in 24 hrs)
    # At least 80% of flow values are non-zero
    candidate_dry_days = (
        df.groupby('date')
        .filter(lambda day: 
            day['wet_weather_event'].eq(0).all() and
            day['missing_data'].eq(0).all() and
            len(day) == 288 and
            (day['flow_lps'] > 0).sum() / len(day) >= 0.8
        )['date']
        .unique()
        .tolist()
    )

    # Filter input dataframe to only include fully dry days
    df_candidate = df[df['timestamp'].dt.date.isin(candidate_dry_days)].copy()

    # Exclude outlier days that do not fit the trend of the rest of the data
    daily_max_flow = df_candidate.groupby('date')['flow_lps'].max()

    _, lower_bound, upper_bound = calculate_IQR(daily_max_flow)

    non_outlier_days = daily_max_flow[(daily_max_flow >= lower_bound) & (daily_max_flow <= upper_bound)].index.tolist()

    df_dry_days = df[df['timestamp'].dt.date.isin(non_outlier_days)].copy()
    return df_dry_days

def _add_time_columns(df):
    df['date'] = df['timestamp'].dt.date
    df['time_of_day'] = df['timestamp'].dt.strftime('%H:%M')
    df['weekday'] = df['timestamp'].dt.strftime('%A')


def calculate_Ro(df_rdii, A, mask):
    """
    Calculates runoff ratio (Ro) as total RDII volume / total rainfall volume
    for periods with wet weather only.
    """

    Q = df_rdii.loc[mask, 'RDII']
    P = df_rdii.loc[mask, 'rainfall_mm']

    # Calculate volume of rain
    V_rain = np.nansum(P) * (1/1000) * A * 10000 * 1000 # Litres

    # Calculate volume of RDII
    V_RDII = np.nansum(Q) * 300 # Litres

    # Calculate runoff ratio
    Ro = V_RDII/V_rain

    return Ro

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

def _calc_base_flow(MDF: float, ADF: float):
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
    conversion_factor = 0.022824465227271 # 0.023mgd = 1L/s

    # Convert lps to mgd
    MDF_mgd = MDF * conversion_factor
    ADF_mgd = ADF * conversion_factor

    # Stevens-Shutzbach Method Equation
    BI_mgd = (0.4 * MDF_mgd) / (1 - 0.6 * ((MDF_mgd / ADF_mgd)**(ADF_mgd**0.7)))
    
    # Convert mgd to lps
    BI = BI_mgd / conversion_factor

    return BI

def _categorize_days(df, separate_fridays):
    """
    Given a dataframe, df, with columns "timestamp"
    return a same dataframe with new column "group" which
    groups days of the week into Workday, Weekend/Holiday, and (optional) Friday.
    
    Also creates columns "date" and "time_of_day" from "timestamp" if they do not already exist.
    """
    df = df.copy()

    # df['date'] = df['timestamp'].dt.date
    # df['time_of_day'] = df['timestamp'].dt.strftime('%H:%M')
    _add_time_columns(df)

    # Get BC holidays for the relevant years
    years = df["timestamp"].dt.year.unique()
    bc_holidays = set(holidays.CA(subdiv = "BC", years = years).keys())

    # Handle separate Friday logic if true
    if separate_fridays == True:
        df.loc[:, "group"] = df["date"].apply(
            lambda x: "Weekend/Holiday" 
            if (x in bc_holidays or x.weekday() >= 5) 
            else "Friday" if x.weekday() == 4  
            else "Workday"
        )
    else:
        df.loc[:, "group"] = df["date"].apply(
            lambda x: "Weekend/Holiday" 
            if (x in bc_holidays or x.weekday() >= 5) 
            else "Workday")
    
    return df

def calculate_diurnal(df_dwf: pd.DataFrame, smooth_window = 3):
    # Calculate median value for each timestep in day
    # Median chosen as less susceptible to outliers
    df_diurnal = df_dwf.groupby(["group", "time_of_day"], as_index = False)["flow_lps"].median()
    df_diurnal["DWF"] = df_diurnal.groupby("group")["flow_lps"].transform(
        lambda x: x.rolling(window=smooth_window, center=True, min_periods=1).mean()
    )    
    
    #df_diurnal.rename(columns = {"flow_lps": "DWF"}, inplace = True)
    df_diurnal["MNF"] = df_diurnal.groupby("group")["DWF"].transform("min") # Minimum Nighttime Flow (MNF)
    df_diurnal["ADWF"] = df_diurnal.groupby("group")["DWF"].transform("mean") # Average Dry Weather Flow (ADWF)
    df_diurnal["GWI"] = df_diurnal.apply(lambda row: _calc_base_flow(row["MNF"], row["ADWF"]), axis = 1) # Ground Water Infiltration
    df_diurnal["SF"] = df_diurnal["DWF"] - df_diurnal["GWI"]

    df_diurnal = df_diurnal.drop(columns=['flow_lps'])

    return df_diurnal

def plot_diurnal(df_diurnal, df_dwf):
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
        subset_data = df_dwf[df_dwf["group"] == group]

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

def decompose_flow(df_flow, df_diurnal, separate_fridays):
    df_flow_grouped = _categorize_days(df_flow, separate_fridays)

    df_comb = pd.merge(
        left = df_flow_grouped,
        right = df_diurnal,
        how = 'left',
        left_on = ['time_of_day', 'group'],
        right_on = ['time_of_day', 'group'],
    )

    # Calculate RDII as Raw Flow - Dry Weather Flow. Floor is 0 L/s.
    df_comb['RDII'] = np.where(df_comb['flow_lps'] - df_comb['DWF'] > 0, df_comb['flow_lps'] - df_comb['DWF'], 0)

    # Drop intermediate calculation columns
    df_comb = df_comb.drop(columns=["time_of_day", "weekday", "group", "date"], errors="ignore")

    return df_comb

def plot_decomposition(df_rdii):
    # Create figure with two rows
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1, 4])

    # Max values for axis scaling
    flow_max = df_rdii["flow_lps"].max()
    rainfall_max = df_rdii["rainfall_mm"].max()

    # Rainfall plot
    fig.add_trace(go.Scatter(
        x=df_rdii["timestamp"],
        y=df_rdii["rainfall_mm"],
        mode="lines",
        name="Rainfall (mm)",
        line=dict(color="darkblue", width=1),
        fill="tozeroy",
        opacity=0.2
    ), row=1, col=1)

    # Flow plot (stacked)
    fig.add_trace(go.Scatter(
        x=df_rdii["timestamp"],
        y=df_rdii["GWI"],
        mode="lines",
        name="GWI (L/s)",
        line=dict(color="red", width=1),
        fill="tozeroy"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_rdii["timestamp"],
        y=df_rdii["DWF"],
        mode="lines",
        name="SF (L/s)",
        line=dict(color="orange", width=1),
        fill="tozeroy"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_rdii["timestamp"],
        y=df_rdii["flow_lps"],
        mode="lines",
        name="RDII (L/s)",
        line=dict(color="green", width=1),
        fill="tonexty"
    ), row=2, col=1)

    # Formatting
    fig.update_layout(
        title="Stacked (Additive) Flow with Rainfall",
        xaxis2=dict(title="Timestamp", tickangle=45),
        yaxis=dict(title="Rainfall (mm)", range=[0, rainfall_max * 1.1]),
        yaxis2=dict(title="Flow (L/s)", range=[0, flow_max * 1.1]),
        legend=dict(title="Legend"),
        template="plotly_white"
    )

    # Show the plot
    fig.show()