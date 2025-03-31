# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback
# %%
# Read in flow data from Flow_Split.py
fp = 'Output_Data/Flow_Categorization.csv'
df_input = (pd.read_csv(fp)
    .assign(timestamp = lambda df: pd.to_datetime(df['timestamp']))
)
# %%
# Create columns for date, time of day, and weekday based on the timestamp column
# These columns are used for later processing of diurnal pattern
df_input['date'] = df_input['timestamp'].dt.date
df_input['time_of_day'] = df_input['timestamp'].dt.strftime('%H:%M')
df_input['weekday'] = df_input['timestamp'].dt.strftime('%A')
# %%
# Identify fully dry days
# A day is considered dry if all 288 5-minute timesteps in a day were previously classified as dry and there is no missing data.
full_dry_days = df_input.groupby(df_input['timestamp'].dt.date).apply(
    lambda g: (
        g['wet_weather_event'].eq(0).all() and 
        g['missing_data'].eq(0).all() and
        len(g) == 288 # Ensure exactly 288 timesteps in the day (# of 5-min time intervals in 24hrs)
        )
).rename('fully_dry')

# Filter input dataframe to only include fully dry days
# This dataframe will form the basis of diurnal pattern calculation
df_dwf = df_input[df_input['timestamp'].dt.date.isin(full_dry_days[full_dry_days].index)]

# %%
# Manual Day Exclusion

# Initialize series which will store the manually selected days
selected_days = pd.Series()

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
            opacity=1,
            customdata=[str(date)] * len(group)  # Store date info
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

app.layout = html.Div([
    dcc.Graph(id="flow-graph", figure=create_figure()),
    html.Div(id="selected-output", style={"margin": "10px 0"}),  
    html.Button("Save Selected Days", id="save-btn", n_clicks=0),  
    html.Div(id="save-output", style={"margin-top": "10px"})  
])

@app.callback(
    [Output("flow-graph", "figure"),
     Output("selected-output", "children")],
    [Input("flow-graph", "clickData")],
    prevent_initial_call=True  
)
def toggle_visibility(click_data):
    """Toggle visibility of the clicked line."""
    fig = create_figure()  

    if click_data:
        clicked_date = click_data["points"][0]["customdata"]  
        for trace in fig.data:
            if trace.name == clicked_date:
                trace.visible = 'legendonly' if trace.visible is True else True

    # Track visible traces
    visible_dates = [trace.name for trace in fig.data if trace.visible is True]

    return fig

@app.callback(
    Output("save-output", "children"),
    [Input("save-btn", "n_clicks")],
    [State("flow-graph", "figure")]
)
def save_selected_days(n_clicks, fig):
    """Store Selected Days in a DataFrame."""
    global selected_days  # Allow modifying the global DataFrame

    if n_clicks > 0:
        visible_dates = [trace["name"] for trace in fig["data"] if trace.get("visible", True) is True]

        # Update the global DataFrame
        selected_days = pd.Series(visible_dates)

    return None

if __name__ == "__main__":
    app.run_server(debug=True)


# %%
# Calculate DWF Diurnal Pattern based on selected dry days from previous step

# Filter flow data based on selected days
df_dwf_filtered = df_dwf[df_dwf["date"].isin(pd.to_datetime(selected_days).dt.date)].copy()

# Get BC holidays for the relevant years
years = df_dwf_filtered["timestamp"].dt.year.unique()
bc_holidays = set(holidays.CA(subdiv = "BC", years = years).keys())

# Classify each fully dry day as either Weekend/Holiday or Workday
# df_dwf_filtered.loc[:, "group"] = df_dwf_filtered["date"].apply(
#     lambda x: "Weekend/Holiday" 
#     if (x in bc_holidays or x.weekday() >= 5) 
#     else "Workday"
# )

df_dwf_filtered.loc[:, "group"] = df_dwf_filtered["date"].apply(
    lambda x: "Weekend/Holiday" 
    if (x in bc_holidays or x.weekday() >= 5) 
    else "Friday" if x.weekday() == 4  
    else "Workday"
)

# Group and calculate mean
df_dwf_diurnal = df_dwf_filtered.groupby(["group", "time_of_day"], as_index = False)["flow_lps"].mean()
# %%
fig, ax = plt.subplots(figsize = (12, 6))

for label, df in df_dwf_diurnal.groupby("group"):
    ax.plot(df["time_of_day"], df["flow_lps"], label = label)

ax.grid()
ax.set_title("Diurnal Patterns")
ax.set_xlabel("Time of Day")
ax.set_ylabel("Flow (L/s)")
plt.xticks(df_dwf_diurnal["time_of_day"].unique()[::24])
plt.legend()
# %%
