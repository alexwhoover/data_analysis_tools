import pandas as pd
import numpy as np
from sewer_analysis.core.raw_data import RawData
from sewer_analysis.core.results_data import ResultsData
from sewer_analysis.analysis.utils import categorize_flow, calculate_diurnal, decompose_flow, plot_diurnal, plot_decomposition, calculate_Ro
from sewer_analysis.apps.storm_selector_app import run_storm_selector_app
from sewer_analysis.analysis.RTK_utils import run_RTK, plot_simulated_flow_dynamic, plot_synthetic_hydrograph

class FlowSite:
    def __init__(self, name: str, catchment_area: float, raw_flow: pd.DataFrame, raw_rainfall: pd.DataFrame, separate_fridays: bool = False):
        self.name = name
        self.catchment_area = catchment_area
        self.raw_data = RawData(raw_flow, raw_rainfall)
        self.results = ResultsData()
        self.separate_fridays = separate_fridays
        
        # A mask for the raw data for wet periods, as determined from categorize_flow()
        # Helpful only for calculating runoff ratio
        self._wet_weather_mask = None 

    def categorize_flow(
            self,
            rolling_window_hr = 6,
            min_intensity_mm_hr = 0.1,
            inter_event_duration_hr = 24,
            min_event_duration_hr = 1,
            response_time_hr = 72,
            lead_time_hr = 2,
            plot = True
    ):
        df_input = self.raw_data.data.copy()
        self.results.dwf_results, self._wet_weather_mask = categorize_flow(
            df_input,
            self.separate_fridays, 
            rolling_window_hr, 
            min_intensity_mm_hr, 
            inter_event_duration_hr,
            min_event_duration_hr,
            response_time_hr,
            lead_time_hr,
            plot = plot
        )

    def calculate_diurnal(self, smooth_window = 3, plot = True):
        self._check_dwf_results_exist()
        df_input = self.results.dwf_results.copy()
        self.results.diurnal_pattern = calculate_diurnal(df_input, smooth_window)

        if plot == True:
            self._plot_diurnal()

    def _plot_diurnal(self):
        self._check_diurnal_results_exist()

        plot_diurnal(self.results.diurnal_pattern, self.results.dwf_results)

    def decompose_flow(self, plot = True):
        self._check_diurnal_results_exist()

        df_flow = self.raw_data.data.copy()
        df_diurnal = self.results.diurnal_pattern.copy()

        self.results.rdii_results = decompose_flow(df_flow, df_diurnal, self.separate_fridays)
        self.results.runoff_ratio = self._calculate_Ro()

        if plot == True:
            self._plot_decomposition()

    def _plot_decomposition(self):
        plot_decomposition(self.results.rdii_results)

    def _calculate_Ro(self):
        self._check_RDII_results_exist()

        # Only calculate runoff ratio for wet weather periods
        Ro = calculate_Ro(self.results.rdii_results, self.catchment_area, self._wet_weather_mask)
        return Ro
    
    def select_RTK_storms(self):
        self._check_RDII_results_exist()
        run_storm_selector_app(self.results.rdii_results)

    def RTK_method(self, pop_size, max_gens):
        try:
            selected_storm_dates_df = pd.read_csv("../data/selected_storm_dates.csv", parse_dates=['start_date', 'end_date'])
        except Exception:
            raise RuntimeError("No selected storm dates csv found")
        
        # Do we want Ro calculated on entire timeseries or just selected periods?
        Ro = self.results.runoff_ratio
        
        res = run_RTK(self.results.rdii_results, selected_storm_dates_df, self.catchment_area, Ro, pop_size, max_gens)
        self.results.RTK_results = res
        
        return res
    
    def print_RTK_values(self):
        res = self.results.RTK_results
        print("RTK1: %s" % res.X[0:3])
        print("RTK2: %s" % res.X[3:6])
        print("RTK3: %s" % res.X[6:9])
    
    def plot_simulated_flow(self, save_file = False):
        self._check_RTK_results_exist()
        
        res = self.results.RTK_results
        plot_simulated_flow_dynamic(self.results.rdii_results, res.X, self.catchment_area, save_file)

    def plot_synthetic_hydrograph(self):
        self._check_RTK_results_exist()

        res = self.results.RTK_results
        plot_synthetic_hydrograph(res.X, R_threshold = 0.001)

    def envelope_method(self):
        pass

    def _check_dwf_results_exist(self):
        if self.results.dwf_results is None:
            raise RuntimeError("You must have DWF results to run this method")
    def _check_diurnal_results_exist(self):
        if self.results.diurnal_pattern is None:
            raise RuntimeError("You must have diurnal pattern results to run this method")
    def _check_RDII_results_exist(self):
        if self.results.rdii_results is None:
            raise RuntimeError("You must have RDII results to run this method")
    def _check_RTK_results_exist(self):
        if self.results.RTK_results is None:
            raise RuntimeError("You must have RTK results to run this method")

