# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import holidays
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# %%
# Options
separate_fridays = False
# %%
# Import data from pre-processing step and DWF pattern
fp1 = 'Input_Data/flow_precip_data.csv'
fp2 = 'Input_Data/DWF_Patterns.csv'
df_flow = (pd.read_csv(fp1)
    .assign(timestamp = lambda df: pd.to_datetime(df['timestamp']))
)
df_diurnal = pd.read_csv(fp2)
# %%
df_flow['date'] = df_flow['timestamp'].dt.date
df_flow['time_of_day'] = df_flow['timestamp'].dt.strftime('%H:%M')
df_flow['weekday'] = df_flow['timestamp'].dt.strftime('%A')

# Get BC holidays for the relevant years
years = df_flow["timestamp"].dt.year.unique()
bc_holidays = set(holidays.CA(subdiv = "BC", years = years).keys())

if separate_fridays == True:
    df_flow.loc[:, "group"] = df_flow["date"].apply(
        lambda x: "Weekend/Holiday" 
        if (x in bc_holidays or x.weekday() >= 5) 
        else "Friday" if x.weekday() == 4  
        else "Workday"
    )
else:
    df_flow.loc[:, "group"] = df_flow["date"].apply(
        lambda x: "Weekend/Holiday" 
        if (x in bc_holidays or x.weekday() >= 5) 
        else "Workday")
# %%
df_comb = pd.merge(
    left = df_flow,
    right = df_diurnal,
    how = 'left',
    left_on = ['time_of_day', 'group'],
    right_on = ['time_of_day', 'group'],
)

df_comb.rename(columns = {"flow_lps_x": "flow_lps", "flow_lps_y": "DWF"}, inplace = True)

df_comb['RDII'] = np.where(df_comb['flow_lps'] - df_comb['DWF'] > 0, df_comb['flow_lps'] - df_comb['DWF'], 0)
df_comb['SF'] = df_comb["DWF"] - df_comb["GWI"]
# %%
# Create figure with two rows
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights = [1, 4])

# Max values for axis scaling
flow_max = df_comb["flow_lps"].max()
rainfall_max = df_comb["rainfall_mm"].max()

# Rainfall plot
fig.add_trace(go.Scatter(
    x=df_comb["timestamp"],
    y=df_comb["rainfall_mm"],
    mode="lines",
    name="Rainfall (mm)",
    line=dict(color="darkblue", width=1),
    fill="tozeroy",
    opacity=0.2
), row=1, col=1)

# Flow plot
fig.add_trace(go.Scatter(
    x=df_comb["timestamp"],
    y=df_comb["flow_lps"],
    mode="lines",
    name="Flow (L/s)",
    line=dict(color="blue", width=1)
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=df_comb["timestamp"],
    y=df_comb["DWF"],
    mode="lines",
    name="DWF (L/s)",
    line=dict(color="orange", width=1)
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=df_comb["timestamp"],
    y=df_comb["RDII"],
    mode="lines",
    name="RDII (L/s)",
    line=dict(color="green", width=1)
), row=2, col=1)

# Formatting
fig.update_layout(
    title="Decomposed Flow with Rainfall",
    xaxis2=dict(title="Timestamp", tickangle=45),
    yaxis=dict(title="Rainfall (mm)", range=[0, rainfall_max * 1.1]),
    yaxis2=dict(title="Flow (L/s)", range=[0, flow_max * 1.1]),
    legend=dict(title="Legend"),
    template="plotly_white"
)

# Show the plot
fig.show()

# %%
# Create figure with two rows
fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1, 4])

# Max values for axis scaling
flow_max = df_comb["flow_lps"].max()
rainfall_max = df_comb["rainfall_mm"].max()

# Rainfall plot
fig2.add_trace(go.Scatter(
    x=df_comb["timestamp"],
    y=df_comb["rainfall_mm"],
    mode="lines",
    name="Rainfall (mm)",
    line=dict(color="darkblue", width=1),
    fill="tozeroy",
    opacity=0.2
), row=1, col=1)

# Flow plot (stacked)


fig2.add_trace(go.Scatter(
    x=df_comb["timestamp"],
    y=df_comb["GWI"],
    mode="lines",
    name="GWI (L/s)",
    line=dict(color="red", width=1),
    fill="tozeroy"
), row=2, col=1)

fig2.add_trace(go.Scatter(
    x=df_comb["timestamp"],
    y=df_comb["DWF"],
    mode="lines",
    name="SF (L/s)",
    line=dict(color="orange", width=1),
    fill="tozeroy"
), row=2, col=1)

fig2.add_trace(go.Scatter(
    x=df_comb["timestamp"],
    y=df_comb["flow_lps"],
    mode="lines",
    name="RDII (L/s)",
    line=dict(color="green", width=1),
    fill="tonexty"
), row=2, col=1)

# Formatting
fig2.update_layout(
    title="Stacked (Additive) Flow with Rainfall",
    xaxis2=dict(title="Timestamp", tickangle=45),
    yaxis=dict(title="Rainfall (mm)", range=[0, rainfall_max * 1.1]),
    yaxis2=dict(title="Flow (L/s)", range=[0, flow_max * 1.1]),
    legend=dict(title="Legend"),
    template="plotly_white"
)

# Show the plot
fig2.show()


# %%
# Save plot and data
pio.write_html(fig, "Output_Data/Flow_Decomposition_Plot.html")
pio.write_html(fig2, "Output_Data/Flow_Decomposition_Stacked_Plot.html")
df_export = df_comb[["timestamp", "depth_mm", "velocity_mps", "flow_lps", "DWF", "RDII", "SF", "GWI"]]
df_export.to_csv("Output_Data/Flow_Decomposition.csv", index = False)
# %%
