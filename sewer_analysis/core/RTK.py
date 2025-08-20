from sewer_analysis.analysis.RTK_utils import RTK, obj_fun_kge
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
import numpy as np
import pandas as pd

def fitness_function(x, A, df_rdii):
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
    Qsim_arrays = []
    for _, group_df in df_rdii.groupby("group"):
        P = group_df["rainfall_mm"].values
        Qsim = RTK(x, P, A, R_threshold = 0.001)
        Qsim_arrays.append(Qsim)


    Qsim = np.concatenate(Qsim_arrays)
    Qobs = df_rdii["RDII"].values

    # Prevents error message if all R values are < 0.05
    # if sum(Qsim) == 0:
    #     return -np.inf

    # Compute Kling Gupta Efficiency Metric
    # Range is (-Inf, 1], where 1 is the optimal number
    # Since the GA is set up as a minimization problem where 0 is optimal, we instead return 1 - KGE
    KGE = obj_fun_kge(Qobs, Qsim)

    return 1 - KGE

class RTKProblem(ElementwiseProblem):

    def __init__(self, A: float, df_input: pd.DataFrame, Ro: float):
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
        
        super().__init__(n_var=9, n_obj=1, n_constr=1, xl = xl, xu = xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Define the model or function to evaluate RMSE
        fitness_score = fitness_function(x, self.A, self.df_input)

        # Set the objective value
        out["F"] = fitness_score # Evaluation Metric: Kling-Gupta Efficiency (KGE)

        Rs = x[0] + x[3] + x[6] # Sum of runoff coefficient (R) values

        out["G"] = Rs - self.Ro # Constraint: Sum of R values should be less than 1

class RTKCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["pop_F"] = []  # Stores the population's objective values
        self.data["pop_X"] = []  # Optionally stores the decision variables

    def notify(self, algorithm):
        # Append the population's objective values for the current generation
        self.data["pop_F"].append(algorithm.pop.get("F").copy())

        # Optionally, append the decision variables for the population
        self.data["pop_X"].append(algorithm.pop.get("X").copy())

