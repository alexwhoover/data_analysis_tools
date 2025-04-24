# %%
# Tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks, peak_prominences
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Dash App
import dash
from dash import html, dcc, Output, Input, State
import plotly.graph_objs as go

# Libraries for genetic algorithm
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultSingleObjectiveTermination

# %%
##############################################################################
##############################################################################
##############################################################################
# TODO: Set Parameters
A = 47.526207  # Set Catchment Area in Hectares
pop_size = 500  # Set initial population size
max_gens = 100  # Set the maximum number of generations before termination
##############################################################################
##############################################################################
##############################################################################


# %%
# Import Data
df_rdii = (pd.read_csv("Input_Data/Flow_Decomposition.csv")
    .assign(timestamp = lambda df: pd.to_datetime(df['timestamp']))
)


# %%
# Storm Selection App
app = dash.Dash(__name__)
server = app.server

# Define App Layout
app.layout = html.Div([
    html.Div([
        html.H1("Define Periods"),

        # Div to choose / display all the storm start and end times
        html.Div(id = "period-container", children = []),

        # Buttons to add / subtract number of storms
        html.Button("Add Period", id = "add-period-btn", n_clicks = 0),
        html.Button("Remove Period", id = "remove-period-btn", n_clicks = 0),
        html.Div(id = "store-periods", style = {"display": "none"})
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

    html.Div([
        dcc.Graph(id = "rainfall-graph")
    ], style={'width': '68%', 'display': 'inline-block', 'padding': '10px'})
])

# Define logic for adding and removing date start / end boxes
@app.callback(
    Output('period-container', 'children'),
    Input('add-period-btn', 'n_clicks'),
    Input('remove-period-btn', 'n_clicks'),
    State({'type': 'period-picker', 'index': dash.ALL}, 'start_date'),
    State({'type': 'period-picker', 'index': dash.ALL}, 'end_date')
)
def update_period_inputs(add_clicks, remove_clicks, start_dates, end_dates):

    # Limit storms to 1 - 10
    delta = add_clicks - remove_clicks + 1
    
    if 1 <= delta <= 10:
        new_count = delta
    elif delta < 1:
        new_count = 1
    elif delta > 10:
        new_count = 10

    # Add / Remove Periods
    new_children = []
    for i in range(new_count):
        start_date = start_dates[i] if i < len(start_dates) else None
        end_date = end_dates[i] if i < len(end_dates) else None
        new_children.extend([
            html.Div([
                dcc.DatePickerRange(
                    id={'type': 'period-picker', 'index': i},
                    min_date_allowed=df_rdii['timestamp'].min().date(),
                    max_date_allowed=df_rdii['timestamp'].max().date(),
                    display_format='YYYY-MM-DD',
                    start_date=start_date,
                    end_date=end_date
                ),
                html.Br()
            ])
        ])
    return new_children

# Define logic for updating graph
@app.callback(
    Output('rainfall-graph', 'figure'),
    Input({'type': 'period-picker', 'index': dash.ALL}, 'start_date'),
    Input({'type': 'period-picker', 'index': dash.ALL}, 'end_date')
)
def update_graph(start_dates, end_dates):
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True)

    fig.add_trace(go.Scatter(
        x=df_rdii["timestamp"],
        y=df_rdii["rainfall_mm"],
        mode="lines",
        name="Rainfall (mm)",
        line=dict(color="darkblue", width=1),
        fill="tozeroy",
        opacity=0.2
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_rdii["timestamp"],
        y=df_rdii["RDII"],
        mode="lines",
        name="RDII (L/s)",
        line=dict(color="green", width=1)
    ), row=2, col=1)

    # Highlight periods on graph
    for i, (start, end) in enumerate(zip(start_dates, end_dates)):
        if start and end:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(255, 0, 0, 0.2)",
                opacity=0.4,
                layer="below",
                line_width=0,
                annotation_text=f"Period {i+1}",
                annotation_position="top left",
            )

    fig.update_layout(
        title="Storm Selector", 
        xaxis_title="Timestamp", 
        yaxis_title="Rainfall (mm)",
        yaxis2_title="RDII (L/s)",
        xaxis2_rangeslider_visible=True, 
        xaxis2_rangeslider_thickness=0.1
    )

    # Save storm dates for later use as a global variable
    global storm_dates
    storm_dates = pd.DataFrame(data = {'start_date': pd.to_datetime(start_dates), 'end_date': pd.to_datetime(end_dates)})

    return fig

# Run App
if __name__ == '__main__':
    app.run_server(debug=True)

# %% [markdown]
# ### Define Functions to Convert RTK & Rainfall to Simulated RDII

# %%
# TODO: Calculation Functions

# Output a unit hydrograph as a numpy array, with each index corresponding to 5-minutes
# Inputs:
# - R (unitless): Runoff Volumetric Coefficient
# - T (hours): Time to Peak
# - K (unitless): Falling Limb Ratio
# Output:
# - unit hydrograph (np.ndarray) with indices 0, 1, 2, ..., i corresponding to 0min, 5min, 10min, ..., T(1+K)
def unit_hydrograph(R, T, K):
    if T <= 0 or K <= 0:
        print("Tp: " + str(T))
        print("K: ", + str(K))
        raise ValueError("Tp and K must be positive non-zero values")

    # Calculation of Qp in units of [L/s] / [mm * ha]
    # RPA is in mm*ha, to convert to L -> [1m/1000mm]*[10000m2/ha]*[1000L/m3] = 10000
    # (T/2)(1+K)Qp in units of [L/s][hrs], to convert to L -> 3600s/hrs
    # Hence, 10 000 / 3600 is to get Qp in units of L/s using precip in mm, catchment area in ha, and T in hours
    Qp = (10000 / 3600) * (2*R) / (T * (1 + K))

    # Generate 5-minute time_steps from 0 to Tmax
    Tmax = T * (1 + K) # Width of unit hydrograph in minutes
    time_steps = np.arange(0, Tmax * 60 + 5, 5) # Create a numpy array of all 5-minute intervals. [0, 5, 10, ..., Tmax]

    # Initialize the empty unit hydrograph
    UH = []

    for t in time_steps: # t is in minutes
        t_hrs = t / 60

        # Rising Limb
        if t_hrs < T:
            qt = Qp * (t_hrs / T)
        # Falling Limb
        elif t_hrs >= T:
            qt = Qp * (1 - (t_hrs - T)/(T * K))
        
        # Using append, so the numpy array is indexed as 0, 1, 2... to represent 0min, 5min, 10min
        # The np.max shouldn't be necessary, but it prevents negative values.
        qt = np.max([qt, 0])

        UH.append(qt)

    return np.array(UH)

# Convolve a 5-minute precipitation time series with a 5-minute unit hydrograph
# Inputs:
# - P (mm): Precipitation time series in millimeters at 5-minute time intervals
# - A (ha): Catchment area in hectares
# - UH (L/s)/(mm-ha): Unit hydrograph derived from unit_hydrograph(R, T, K)
def transform_rainfall_with_UH(P, A, UH):
    # Multiply rainfall by catchment area to get rainfall volume in mm-ha
    P_scaled = P * A

    # Apply convolution transformation on rainfall volume and unit hydrograph to get simulated flow in L/s
    simulated_flow = np.convolve(P_scaled, UH, mode = 'full')[:len(P)]

    return simulated_flow

# Derive simulated RDII hydrograph from RTK parameters, rainfall, and catchment area
# Inputs:
# - x = [R1, T1, K1, R2, T2, K2, R3, T3, K3] is the RTK parameters
# - P (mm): Precipitation time series in millimeters at 5-minute time intervals
# - A (ha): Catchment area in hectares 
def RTK(x, P, A, R_threshold = 0.05):
    # Fill all nan values in precip time series with 0
    # This assumes there are no major gaps in precip time series
    P_filled = np.where(np.isnan(P), 0.0, P)
    simulated_flow = np.zeros_like(P_filled)

    R1, T1, K1, R2, T2, K2, R3, T3, K3 = x

    if R1 >= R_threshold:
        UH1 = unit_hydrograph(R1, T1, K1)
        Q1 = transform_rainfall_with_UH(P_filled, A, UH1)
        simulated_flow += Q1

    if R2 >= R_threshold:
        UH2 = unit_hydrograph(R2, T2, K2)
        Q2 = transform_rainfall_with_UH(P_filled, A, UH2)
        simulated_flow += Q2

    if R3 >= R_threshold:
        UH3 = unit_hydrograph(R3, T3, K3)
        Q3 = transform_rainfall_with_UH(P_filled, A, UH3)
        simulated_flow += Q3

    return simulated_flow

# def find_storms(Q, prominence, num_peaks):

#     # Find all peaks with at least prominence = prominence
#     peaks, _ = find_peaks(Q, prominence = prominence)

#     # Get prominence values for all peaks found
#     prominences = peak_prominences(Q, peaks)[0]

#     # Sort the peaks by prominence, descending
#     top_indices = sorted(range(len(prominences)), key=lambda i: prominences[i], reverse=True)[:num_peaks]

#     # Get the actual indices in Q of the [num_peaks] most prominent peaks
#     top_peaks = peaks[top_indices]

#     return top_peaks

# def calculate_flow_events(df_input, min_intensity_mm_hr = 0.1, rolling_window_hr = 6, response_time_hr = 24, inter_event_duration_hr = 24, min_event_duration_hr = 1, lead_time_hr = 2):
#     window_size = int(rolling_window_hr * 60 / 5)

#     # Calculate rolling sum of precip, then convert to intensity
#     # Duration of intensity calculation = rolling_window_hr
#     df_input['rainfall_roll_sum'] = df_input['rainfall_mm'].rolling(window = window_size).sum()
#     df_input['rainfall_roll_sum_intensity'] = df_input['rainfall_roll_sum'] / rolling_window_hr

#     # Initialize the wet_weather_event column with 0
#     df_input['wet_weather_event'] = 0

#     # Identify potential events based on intensity threshold
#     # event_indices is a list of all indices with precip >= threshold
#     event_indices = df_input.index[
#         df_input['rainfall_roll_sum_intensity'] >= min_intensity_mm_hr
#     ].tolist()

#     # If there are no rainfall events, return df_input
#     if not event_indices:
#         return df_input
    
#     # Combine events within the inter-event duration using timestamps
#     current_event_start = df_input.loc[event_indices[0], 'timestamp']
#     current_event_end = df_input.loc[event_indices[0], 'timestamp']

#     for i in range(1, len(event_indices)):
#         current_timestamp = df_input.loc[event_indices[i], 'timestamp']

#         # If two timestamps above threshold are within inter-event duration, change event end
#         if (current_timestamp - current_event_end).total_seconds() <= (inter_event_duration_hr * 3600):
#             current_event_end = current_timestamp

#         # Once the next timestamp above threshold is outside of inter-event duration, check if event duration meets minimum requirement
#         elif (current_event_end - current_event_start).total_seconds() >= (min_event_duration_hr * 3600):
#             # If minimum requirement met, mark as storm
#             df_input.loc[
#                 (df_input['timestamp'] >= (current_event_start - timedelta(hours = lead_time_hr))) & (df_input['timestamp'] <= (current_event_end + timedelta(hours = response_time_hr))),
#                 'wet_weather_event'
#             ] = 1

#             current_event_start = current_timestamp
#             current_event_end = current_timestamp

#     # Add a column showing periods with NA values in flow or precip
#     df_input['missing_data'] = (~((~df_input['rainfall_mm'].isnull()) & (~df_input['flow_lps'].isnull()))).astype(int)
    
#     return df_input

# %%
# Fitness Function
# This function calculates the Kling Gupta Efficiency (KGE) used to assess model performance.
# Credit goes to Jean-Luc Martel and Richard Arsenault
def obj_fun_kge(Qobs, Qsim):
    """
    obj_fun_kge

    Returns the modified KGE as proposed by Gupta et al. (2009) and
    modified by Kling et al. (2012).

    INPUTS :
        Qobs : Vector of observed streamflow (m3 s-1) [n x 1]
        Qsim : Vector of simulated flow (m3 s-1) [n x 1]

    OUTPUT :
        kge : Kling-Gupta Efficiency (KGE) metric

    REFERENCES :
        - Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
          Decomposition of the mean squared error and NSE performance
          criteria: Implications for improving hydrological modelling. Journal
          of Hydrology, 377(1), 80-91. doi:10.1016/j.jhydrol.2009.08.003
        - Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the
          upper Danube basin under an ensemble of climate change scenarios.
          Journal of Hydrology, 424-425, 264-277.
          doi:10.1016/j.jhydrol.2012.01.011
    """

    # Delete all nan values from observed and simulated flows
    ind_nan = np.isnan(Qobs)
    Qobs = Qobs[~ind_nan]
    Qsim = Qsim[~ind_nan]

    # Calculate the correlation coefficient (r)
    r = np.corrcoef(Qsim, Qobs)[0, 1]

    # Calculate the bias ratio b (beta)
    b = (np.mean(Qsim) / np.mean(Qobs))

    # Calculate the variability ratio g (gamma)
    g = (np.std(Qsim) / np.mean(Qsim)) / (np.std(Qobs) / np.mean(Qobs))

    # Calculate the modified KGE
    kge = 1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)

    # In some instances, the results could return NaN. To avoid numerical errors
    # we instead return negative infinite that can let an optimizer understand
    # the direction of the loss.
    if np.isnan(kge):
        kge = -np.inf

    return kge

# Calculate the root mean squared error between an observed time series and simulated time series
# Not used in training process, however, kept here in case one wants to compare KGE with RMSE results
# Inputs:
# - Qobs (numpy array): observed time series
# - Qsim (numpy array): simulated time series
# Output:
# - RMSE
def obj_fun_rmse(Qobs, Qsim):
    # Delete all nan values from observed and simulated flows
    ind_nan = np.isnan(Qobs)
    Qobs = Qobs[~ind_nan]
    Qsim = Qsim[~ind_nan]

    RMSE = np.sqrt(
        np.mean((Qobs - Qsim) ** 2)
    )

    return RMSE

# Define fitness function to evaulate solution performance during model training
# This function takes a solution, x, then converts that to a simulated flow series. The simulated flow series is then compared to the actual flow series using the KGE metric.
# Due to the range of KGE being (-inf, 1], the output is transformed to 1 - KGE to change the range to [0, Inf), which is better suited to the genetic algorithm package pymoo.
# Inputs:
# - x = [R1, T1, K1, R2, T2, K2, R3, T3, K3] is the RTK parameters
# - P (mm): Precipitation time series in millimeters at 5-minute time intervals
# - A (ha): Catchment area in hectares
# - Qobs (numpy array): Observed time series
# Outputs:
# - (1 - KGE) (float): A metric to evaluate solution performance during training

def fitness_function(x, A, df_input, mask):
    Qsim_arrays = []
    for _, group_df in df_input.groupby("group"):
        P = group_df["rainfall_mm"].values
        Qsim = RTK(x, P, A, R_threshold = 0.001)
        Qsim_arrays.append(Qsim)


    Qsim = np.concatenate(Qsim_arrays)
    Qobs = df_input["RDII"].values

    # Prevents error message if all R values are < 0.05
    # if sum(Qsim) == 0:
    #     return -np.inf

    # Compute Kling Gupta Efficiency Metric
    # Range is (-Inf, 1], where 1 is the optimal number
    # Since the GA is set up as a minimization problem where 0 is optimal, we instead return 1 - KGE
    KGE = obj_fun_kge(Qobs, Qsim)

    return 1 - KGE
    
# Plotting Functions
# Plot Synthetic Hydrograph
# Inputs:
# - x = [R1, T1, K1, R2, T2, K2, R3, T3, K3] is the RTK parameters
# Output:
# - None
# def plot_synthetic_hydrograph(x):
#     R1, T1, K1, R2, T2, K2, R3, T3, K3 = x

#     UH1 = unit_hydrograph(R1, T1, K1)
#     UH2 = unit_hydrograph(R2, T2, K2)
#     UH3 = unit_hydrograph(R3, T3, K3)

#     max_length = max(len(UH1), len(UH2), len(UH3))
#     UH1_padded = np.pad(UH1, (0, max_length - len(UH1)))
#     UH2_padded = np.pad(UH2, (0, max_length - len(UH2)))
#     UH3_padded = np.pad(UH3, (0, max_length - len(UH3)))

#     hydrograph = UH1_padded + UH2_padded + UH3_padded

#     fig, ax = plt.subplots(figsize = (8, 6))

#     ax.plot(np.arange(0, max_length * 5, 5), hydrograph, alpha = 1, color = 'black', label = "Synthetic RTK Hydrograph")
#     ax.plot(np.arange(0, len(UH1) * 5, 5), UH1, alpha = 1, color = 'r')
#     ax.plot(np.arange(0, len(UH2) * 5, 5), UH2, alpha = 1, color = 'g')
#     ax.plot(np.arange(0, len(UH3) * 5, 5), UH3, alpha = 1, color = 'b')
#     ax.set_xlabel("Time (minutes)")
#     ax.set_ylabel("Flow (L/s)/(mm-ha)")
#     ax.grid()

#     plt.savefig('Output_Data/RTK_Unit_Hydrograph.png')

#     return None

def plot_synthetic_hydrograph(x, R_threshold = 0.001):
    R1, T1, K1, R2, T2, K2, R3, T3, K3 = x

    hydrographs = []
    colors = []
    labels = []

    if R1 >= R_threshold:
        UH1 = unit_hydrograph(R1, T1, K1)
        hydrographs.append(UH1)
        colors.append('r')
        labels.append('UH1')

    if R2 >= R_threshold:
        UH2 = unit_hydrograph(R2, T2, K2)
        hydrographs.append(UH2)
        colors.append('g')
        labels.append('UH2')

    if R3 >= R_threshold:
        UH3 = unit_hydrograph(R3, T3, K3)
        hydrographs.append(UH3)
        colors.append('b')
        labels.append('UH3')

    if not hydrographs:
        print("No unit hydrographs above the threshold.")
        return None

    max_length = max(len(uh) for uh in hydrographs)
    padded_hydrographs = [np.pad(uh, (0, max_length - len(uh))) for uh in hydrographs]

    synthetic_hydrograph = np.sum(padded_hydrographs, axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))

    for uh, color, label in zip(hydrographs, colors, labels):
        ax.fill(np.arange(0, len(uh) * 5, 5), uh, color=color, label=label)

    ax.plot(np.arange(0, max_length * 5, 5), synthetic_hydrograph, color='black', label="Synthetic RTK Hydrograph", lw = 3)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Flow (L/s)/(mm-ha)")
    ax.grid()
    ax.legend()

    plt.savefig('Output_Data/RTK_Unit_Hydrograph.png')
    return None

# Highlight the local maxima of a time-series, Q, in a plot. Only local maxima of a certain prominence are shown. Prominence is a measure of peak magnitude relative to nearby peaks.
# Inputs:
# - Q: Time-series
# - prominence (float): 
# def plot_peaks(Q, peak_indices):
#     plt.plot(Q)
#     plt.plot(peak_indices, Q[peak_indices], "x")
#     plt.show()

# def plot_simulated_flow(x, P, A, Q, timestamp):
#     Q_sim = RTK(x, P, A)

#     # Plot simulated flow
#     fig, ax = plt.subplots(figsize = (8, 6))

#     ax.plot(timestamp, Q, alpha = 0.7, color = 'b', label = "Actual RDII")
#     ax.plot(timestamp, Q_sim, alpha = 0.7, color = 'r', label = "Simulated RDII")
#     ax.set_xlabel("Time (min)")
#     ax.set_ylabel("Flow (L/s)")
#     ax.set_title("Actual vs. Simulated Flow")
#     ax.legend()

#     return None

def plot_simulated_flow_dynamic(x, P, A, Q, timestamp, save_file = False):
    Q_sim = RTK(x, P, A, R_threshold = 0.001)

    # Create the figure
    #fig = go.Figure()

    # Create subplots with shared x-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=("Actual vs. Simulated Flow", "Precipitation"))

    # Add Actual RDII
    fig.add_trace(go.Scatter(
        x=timestamp,
        y=Q,
        mode='lines',
        name='Actual RDII',
        line=dict(color='green', width=2),
        opacity=1
    ), row = 1, col = 1)

    # Add Simulated RDII
    fig.add_trace(go.Scatter(
        x=timestamp,
        y=Q_sim,
        mode='lines',
        name='Simulated RDII',
        line=dict(color='red', width=2),
        opacity=1
    ), row = 1, col = 1)

    # Add Precipitation in different plot
    fig.add_trace(go.Scatter(
        x=timestamp,
        y=P,
        mode='lines',
        name='Precipitation',
        line=dict(color='blue', width = 2),
        opacity = 1
    ), row=2, col=1)


    fig.update_layout(
        xaxis2_title="Time (min)",
        yaxis_title="Flow (L/s)",
        yaxis2=dict(title="Precipitation (mm)"),  # Second plot y-axis title
        template="plotly_white"
    )

    fig.show()
    
    if save_file == True:
        pio.write_html(fig, file="Output_Data/Comparison_Plot.html", auto_open=False)

    return None


# %%
# Define the problem class
class MyProblem(ElementwiseProblem):

    def __init__(self, A: float, df_input: pd.DataFrame, Ro: float, mask: pd.Series):
        # Define parameter bounds
        R_bounds = (0, 1)
        T_bounds = (0, 24)
        K_bounds = (0, 10)
        
        # Combine bounds for all variables
        xl = np.array([R_bounds[0], T_bounds[0], K_bounds[0], R_bounds[0], T_bounds[0], K_bounds[0], R_bounds[0], T_bounds[0], K_bounds[0]])
        xu = np.array([R_bounds[1], T_bounds[1], K_bounds[1], R_bounds[1], T_bounds[1], K_bounds[1], R_bounds[1], T_bounds[1], K_bounds[1]])

        self.df_input = df_input
        self.A = A
        self.Ro = Ro
        self.mask = mask
        
        super().__init__(n_var=9, n_obj=1, n_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Define the model or function to evaluate RMSE
        fitness_score = fitness_function(x, self.A, self.df_input, self.mask)

        # Set the objective value
        out["F"] = fitness_score # Evaluation Metric: Kling-Gupta Efficiency (KGE)

        Rs = x[0] + x[3] + x[6] # Sum of runoff coefficient (R) values

        out["G"] = Rs - self.Ro # Constraint: Sum of R values should be less than 1


# %%
# Define callback class to save intermediate results for later visualization
class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["pop_F"] = []  # Stores the population's objective values
        self.data["pop_X"] = []  # Optionally stores the decision variables

    def notify(self, algorithm):
        # Append the population's objective values for the current generation
        self.data["pop_F"].append(algorithm.pop.get("F").copy())

        # Optionally, append the decision variables for the population
        self.data["pop_X"].append(algorithm.pop.get("X").copy())



# %%
# Do some preliminary calculations and formatting for gen alg and output stats
df_input = df_rdii.loc[:, ['timestamp', 'rainfall_mm', 'RDII']] # Set a dataframe with columns timestamp, rainfall_mm, and RDII
Q = df_input['RDII']
P = df_input['rainfall_mm']

# Calculate volume of rain
V_rain = np.nansum(P) * (1/1000) * A * 10000 * 1000 # Litres

# Calculate volume of RDII
V_RDII = np.nansum(Q) * 300 # Litres

# Calculate runoff ratio
Ro = V_RDII/V_rain

# Calculate mask for storms
df_input_filtered = pd.DataFrame()

for i, row in storm_dates.iterrows():
    mask = (df_input['timestamp'] >= row['start_date']) & (df_input['timestamp'] <= row['end_date'])
    df_temp = df_input.loc[mask].copy()
    df_temp['group'] = i + 1
    df_input_filtered = pd.concat([df_input_filtered, df_temp], ignore_index = True)
# %%
# Initialize Problem
# Create an instance of the problem
problem = MyProblem(
    A = A,
    df_input = df_input_filtered,
    Ro = Ro,
    mask = mask
)

# Define the genetic algorithm
algorithm = GA(
    pop_size = pop_size,
    sampling = FloatRandomSampling(),
    crossover = SBX(prob=0.9, eta=15),
    mutation = PM(eta=20),
    eliminate_duplicates = True,
)

termination = DefaultSingleObjectiveTermination(
    ftol = 1e-4,
    period = 10, 
    n_max_gen = max_gens
)

# %%
# TODO: Optimize
# Perform the optimization
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True,
               callback = MyCallback())

# %%
# Export Results
Q_sim = RTK(res.X, P, A, R_threshold = 0.001)
ind_nan = np.isnan(Q) | np.isnan(Q_sim)  # Mask NaNs in either Q or Q_sim
Q_tot = np.sum(Q[~ind_nan])
Q_sim_tot = np.sum(Q_sim[~ind_nan])
Volume_Ratio = Q_tot / Q_sim_tot
R_Ratio = Ro / (res.X[0] + res.X[3] + res.X[6])

# Print the best solution found
f = open("Output_Data/RTK_Results.txt", "x")

f.write("RTK1: %s\n" % res.X[0:3])
f.write("RTK2: %s\n" % res.X[3:6])
f.write("RTK3: %s\n" % res.X[6:9])
f.write("KGE: %.4f\n" % obj_fun_kge(Q, Q_sim))
f.write("RMSE: %.4f\n" % obj_fun_rmse(Q, Q_sim))
f.write("Total Flow / Total Simulated Flow: %.4f\n" % Volume_Ratio)
f.write("Actual Runoff Ratio / Simulated Runoff Ratio: %.4f\n" % R_Ratio)
f.close()

# %%
# TODO: Plot Best Solution

plot_synthetic_hydrograph(res.X)
plot_simulated_flow_dynamic(res.X, P, A, Q, np.array(df_rdii['timestamp']), save_file = True)

# %%
