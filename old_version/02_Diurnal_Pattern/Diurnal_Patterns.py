# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback

# %%
# Options ####
separate_fridays = True # Option to make Friday its own group in the diurnal calculations
# %%
# Functions
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

# %%
# Read in flow data from Flow_Categorization.py
fp = 'Output_Data/flow_categorization.csv'
df_input = (pd.read_csv(fp)
    .assign(timestamp = lambda df: pd.to_datetime(df['timestamp']))
)
# %%
# Data Preprocessing

# Create columns for date, time of day, and weekday based on the timestamp column
# These columns are used for later processing of diurnal pattern
df_input['date'] = df_input['timestamp'].dt.date
df_input['time_of_day'] = df_input['timestamp'].dt.strftime('%H:%M')
df_input['weekday'] = df_input['timestamp'].dt.strftime('%A')
# %%
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

# %%
# Manual Day Exclusion

# Create Dash app which displays all the fully dry days in df_dwf, then allows the user to manually deselect certain days from further calculations
app = Dash(__name__)

# Function to generate the initial figure
def create_figure():
    fig = go.Figure()
    for date, group in df_dwf.groupby("date"):
        fig.add_trace(go.Scatter(
            x=group["time_of_day"],
            y=group["flow_lps"],
            mode='lines',
            name=str(date),
            line=dict(width=1),
            opacity=1
        ))

    fig.update_layout(
        title="Flow for Fully Dry Days",
        xaxis_title="Time of Day",
        yaxis_title="Flow (L/s)",
        xaxis=dict(tickvals=df_dwf["time_of_day"].unique()[::12], tickangle=45),
        legend_title="Date",
        template = "plotly_white"
    )
    return fig

# Define the layout of the Dash app
app.layout = html.Div([
    # Graph
    dcc.Graph(id="flow-graph", figure=create_figure()),

    # Button
    html.Button("Save Selected Days", id="save-btn", n_clicks=0), 

    # Confirmation Message 
    html.Div(id="save-output", style={"margin-top": "10px"})
],
    style = {"backgroundColor": "white"}
)

@app.callback(
    Output("save-output", "children"), # Send confirmation message to output
    [Input("save-btn", "n_clicks")], # Triggered when button is clicked
    [State("flow-graph", "figure")] # Reads the figure state to get visible traces
)
def save_selected_days(n_clicks, fig):
    """Store Selected Days in a DataFrame."""
    global selected_days  # Allow modifying the global DataFrame

    if n_clicks > 0:
        # Get all currently visible dates
        visible_dates = [trace["name"] for trace in fig["data"] if trace.get("visible", True) is True]

        # Update the global DataFrame
        selected_days = pd.Series(visible_dates)

        # Confirmation message
        return "Saved selected days."
    return "Days not yet saved."

if __name__ == "__main__":
    app.run_server(debug=True)


# %%
# Calculate DWF Diurnal Pattern based on selected dry days from previous step

# Filter flow data based on selected days
df_dwf_filtered = df_dwf[df_dwf["date"].isin(pd.to_datetime(selected_days).dt.date)].copy()

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
df_dwf_diurnal = df_dwf_filtered.groupby(["group", "time_of_day"], as_index = False)["flow_lps"].mean()
df_dwf_diurnal.rename(columns = {"flow_lps": "DWF"}, inplace = True)
df_dwf_diurnal["MNF"] = df_dwf_diurnal.groupby("group")["DWF"].transform("min") # Minimum Nighttime Flow (MNF)
df_dwf_diurnal["ADWF"] = df_dwf_diurnal.groupby("group")["DWF"].transform("mean") # Average Dry Weather Flow (ADWF)
df_dwf_diurnal["GWI"] = df_dwf_diurnal.apply(lambda row: calc_base_flow(row["MNF"], row["ADWF"]), axis = 1) # Ground Water Infiltration
df_dwf_diurnal["SF"] = df_dwf_diurnal["DWF"] - df_dwf_diurnal["GWI"]

# %%
# Get unique groups
groups = df_dwf_diurnal["group"].unique()

# Create figure and subplots
fig, axes = plt.subplots(nrows=1, ncols=len(groups), figsize=(16, 9), sharey=True)

# Ensure axes is iterable even if there's only one group
if len(groups) == 1:
    axes = [axes]

# Loop through each group and plot
for ax, group in zip(axes, groups):
    subset_diurnal = df_dwf_diurnal[df_dwf_diurnal["group"] == group]
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

# %%
# Save diurnal patterns to csv
df_dwf_diurnal.to_csv("Output_Data/DWF_Patterns.csv", index = False)
# %%
