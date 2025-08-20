import numpy as np
import pandas as pd

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultSingleObjectiveTermination

def unit_hydrograph(R, T, K):
    # Output a unit hydrograph as a numpy array, with each index corresponding to 5-minutes
    # Inputs:
    # - R (unitless): Runoff Volumetric Coefficient
    # - T (hours): Time to Peak
    # - K (unitless): Falling Limb Ratio
    # Output:
    # - unit hydrograph (np.ndarray) with indices 0, 1, 2, ..., i corresponding to 0min, 5min, 10min, ..., T(1+K)
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

def transform_rainfall_with_UH(P, A, UH):
    # Convolve a 5-minute precipitation time series with a 5-minute unit hydrograph
    # Inputs:
    # - P (mm): Precipitation time series in millimeters at 5-minute time intervals
    # - A (ha): Catchment area in hectares
    # - UH (L/s)/(mm-ha): Unit hydrograph derived from unit_hydrograph(R, T, K)
    # Multiply rainfall by catchment area to get rainfall volume in mm-ha
    P_scaled = P * A

    # Apply convolution transformation on rainfall volume and unit hydrograph to get simulated flow in L/s
    simulated_flow = np.convolve(P_scaled, UH, mode = 'full')[:len(P)]

    return simulated_flow

def RTK(x, P, A, R_threshold = 0.05):
    # Derive simulated RDII hydrograph from RTK parameters, rainfall, and catchment area
    # Inputs:
    # - x = [R1, T1, K1, R2, T2, K2, R3, T3, K3] is the RTK parameters
    # - P (mm): Precipitation time series in millimeters at 5-minute time intervals
    # - A (ha): Catchment area in hectares 

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


def obj_fun_rmse(Qobs, Qsim):
    # Calculate the root mean squared error between an observed time series and simulated time series
    # Not used in training process, however, kept here in case one wants to compare KGE with RMSE results
    # Inputs:
    # - Qobs (numpy array): observed time series
    # - Qsim (numpy array): simulated time series
    # Output:
    # - RMSE
    # Delete all nan values from observed and simulated flows
    ind_nan = np.isnan(Qobs)
    Qobs = Qobs[~ind_nan]
    Qsim = Qsim[~ind_nan]

    RMSE = np.sqrt(
        np.mean((Qobs - Qsim) ** 2)
    )

    return RMSE

def run_RTK(df_rdii, selected_storm_dates, A, Ro, pop_size, max_gens):
    from sewer_analysis.core.RTK import RTKProblem, RTKCallback
    
    # Convert start_date and end_date columns to datetime.date objects
    selected_storm_dates['start_date'] = pd.to_datetime(selected_storm_dates['start_date'])
    selected_storm_dates['end_date'] = pd.to_datetime(selected_storm_dates['end_date'])

    filtered_dfs = []
    for i, row in selected_storm_dates.iterrows():
        mask = (df_rdii['timestamp'] >= row['start_date']) & (df_rdii['timestamp'] <= row['end_date'])
        df_temp = df_rdii.loc[mask].copy()
        df_temp['group'] = i + 1
        filtered_dfs.append(df_temp)

    df_rdii_filtered = pd.concat(filtered_dfs, ignore_index=True)


    rtk_problem = RTKProblem(A, df_rdii_filtered, Ro)
    
    algorithm = GA(
        pop_size = pop_size,
        sampling = FloatRandomSampling(),
        crossover = SBX(prob=0.9, eta=15),
        mutation = PM(eta=20),
        eliminate_duplicates = True
    )

    termination = DefaultSingleObjectiveTermination(
        ftol = 1e-4,
        period = 10, 
        n_max_gen = max_gens
    )

    # Perform the optimization
    res = minimize(rtk_problem,
                algorithm,
                termination,
                seed=1,
                verbose=True,
                callback = RTKCallback())
