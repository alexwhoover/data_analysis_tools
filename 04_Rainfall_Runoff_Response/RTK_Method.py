# %%
# Tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultSingleObjectiveTermination


# %% [markdown]
# ### Data Import and Exploration
# %%
# Import Sample Data for ANG_COM002
# TODO: Import Data
df_rdii = pd.read_csv("Inputs/BLC_STM013.csv")

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
def RTK(x, P, A):
    # Fill all nan values in precip time series with 0
    # This assumes there are no major gaps in precip time series
    P_filled = np.where(np.isnan(P), 0.0, P)

    R1, T1, K1, R2, T2, K2, R3, T3, K3 = x

    UH1 = unit_hydrograph(R1, T1, K1)
    UH2 = unit_hydrograph(R2, T2, K2)
    UH3 = unit_hydrograph(R3, T3, K3)

    Q1 = transform_rainfall_with_UH(P_filled, A, UH1)
    Q2 = transform_rainfall_with_UH(P_filled, A, UH2)
    Q3 = transform_rainfall_with_UH(P_filled, A, UH3)

    simulated_flow = np.add(Q1, Q2, Q3)

    return simulated_flow



# %%
# TODO: Fitness Function
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

def fitness_function(x, P, A, Qobs):
    # Calculate Simulated Flow
    Qsim = RTK(x, P, A)

    # Compute Kling Gupta Efficiency Metric
    # Range is (-Inf, 1], where 1 is the optimal number
    # Since the GA is set up as a minimization problem where 0 is optimal, we instead return 1 - KGE
    KGE = obj_fun_kge(Qobs, Qsim)

    return 1 - KGE
    
# TODO: Plotting Functions
# Plot Synthetic Hydrograph
# Inputs:
# - x = [R1, T1, K1, R2, T2, K2, R3, T3, K3] is the RTK parameters
# Output:
# - None
def plot_synthetic_hydrograph(x):
    R1, T1, K1, R2, T2, K2, R3, T3, K3 = x

    UH1 = unit_hydrograph(R1, T1, K1)
    UH2 = unit_hydrograph(R2, T2, K2)
    UH3 = unit_hydrograph(R3, T3, K3)

    max_length = max(len(UH1), len(UH2), len(UH3))
    UH1_padded = np.pad(UH1, (0, max_length - len(UH1)))
    UH2_padded = np.pad(UH2, (0, max_length - len(UH2)))
    UH3_padded = np.pad(UH3, (0, max_length - len(UH3)))

    hydrograph = UH1_padded + UH2_padded + UH3_padded

    fig, ax = plt.subplots(figsize = (8, 6))

    ax.plot(np.arange(0, max_length * 5, 5), hydrograph, alpha = 1, color = 'black', label = "Synthetic RTK Hydrograph")
    ax.plot(np.arange(0, len(UH1) * 5, 5), UH1, alpha = 1, color = 'r')
    ax.plot(np.arange(0, len(UH2) * 5, 5), UH2, alpha = 1, color = 'g')
    ax.plot(np.arange(0, len(UH3) * 5, 5), UH3, alpha = 1, color = 'b')
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Flow (L/s)/(mm-ha)")

    return None

# Highlight the local maxima of a time-series, Q, in a plot. Only local maxima of a certain prominence are shown. Prominence is a measure of peak magnitude relative to nearby peaks.
# Inputs:
# - Q: Time-series
# - prominence (float): 
def plot_peaks(Q, prominence):
    peaks, _ = find_peaks(Q, prominence = prominence)

    plt.plot(Q)
    plt.plot(peaks, Q[peaks], "x")
    plt.show()

def plot_simulated_flow(x, P, A, Q, timestamp):
    Q_sim = RTK(x, P, A)

    # Plot simulated flow
    fig, ax = plt.subplots(figsize = (8, 6))

    ax.plot(timestamp, Q, alpha = 0.7, color = 'b', label = "Actual RDII")
    ax.plot(timestamp, Q_sim, alpha = 0.7, color = 'r', label = "Simulated RDII")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Flow (L/s)")
    ax.set_title("Actual vs. Simulated Flow")
    ax.legend()

    return None

def plot_simulated_flow_dynamic(x, P, A, Q, timestamp, save_file = False):
    Q_sim = RTK(x, P, A)

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
        xaxis_title="Time (min)",
        yaxis_title="Flow (L/s)",
        yaxis2=dict(title="Precipitation (mm)"),  # Second plot y-axis title
        template="plotly_white"
    )

    fig.show()
    
    if save_file == True:
        pio.write_html(fig, file="plot.html", auto_open=True)

    return None


# %%
# TODO: Define the problem class
class MyProblem(ElementwiseProblem):

    def __init__(self, P: np.ndarray, A: float, Q: np.ndarray, Ro: float):
        # Define parameter bounds
        R_bounds = (0, 1)
        T_bounds = (0, 24)
        K_bounds = (0, 10)
        
        # Combine bounds for all variables
        xl = np.array([R_bounds[0], T_bounds[0], K_bounds[0], R_bounds[0], T_bounds[0], K_bounds[0], R_bounds[0], T_bounds[0], K_bounds[0]])
        xu = np.array([R_bounds[1], T_bounds[1], K_bounds[1], R_bounds[1], T_bounds[1], K_bounds[1], R_bounds[1], T_bounds[1], K_bounds[1]])

        self.P = P
        self.A = A
        self.Q = Q
        self.Ro = Ro
        
        super().__init__(n_var=9, n_obj=1, n_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Define the model or function to evaluate RMSE
        fitness_score = fitness_function(x, self.P, self.A, self.Q)

        # Set the objective value
        out["F"] = fitness_score

        Rs = x[0] + x[3] + x[6] # Sum of R values

        out["G"] = Rs - self.Ro # Sum of R values should be less than 1


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
# TODO: Set Parameters
A = 10.525275  # ha
P = np.array(df_rdii['rainfall_mm'])
Q = np.array(df_rdii['RDII_lps'])
V_rain = np.nansum(P) * (1/1000) * A * 10000 * 1000 # Litres
V_RDII = np.nansum(Q) * 300 # Litres
Ro = V_RDII/V_rain
timestamp = np.array(df_rdii['timestamp'])
prominence = 50
weight = 10
pop_size = 500
max_gens = 200

# %%
# TODO: Initialize Problem
# Create an instance of the problem
problem = MyProblem(
    P = P,
    A = A,
    Q = Q,
    Ro = Ro
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
Q_sim = RTK(res.X, P, A)
ind_nan = np.isnan(Q) | np.isnan(Q_sim)  # Mask NaNs in either Q or Q_sim
Q_tot = np.sum(Q[~ind_nan])
Q_sim_tot = np.sum(Q_sim[~ind_nan])
Volume_Ratio = Q_tot / Q_sim_tot
R_Ratio = Ro / (res.X[0] + res.X[3] + res.X[6])

# Print the best solution found
print("RTK1: %s" % res.X[0:3])
print("RTK2: %s" % res.X[3:6])
print("RTK3: %s" % res.X[6:9])
print("Function value: %.4f" % res.F[0])
print("KGE: %.4f" % obj_fun_kge(Q, Q_sim))
print("RMSE: %.4f" % obj_fun_rmse(Q, Q_sim))
print("Total Flow / Total Simulated Flow: %.4f" % Volume_Ratio)
print("Actual Runoff Ratio / Simulated Runoff Ratio: %.4f" % R_Ratio)

# %%
# TODO: Plot Best Solution

plot_synthetic_hydrograph(res.X)
plot_simulated_flow_dynamic(res.X, P, A, Q, timestamp, save_file = True)

# %%
