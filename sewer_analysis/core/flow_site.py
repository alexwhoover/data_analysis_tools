import pandas as pd
import numpy as np
from sewer_analysis.core.raw_data import RawData
from sewer_analysis.core.results_data import ResultsData
from sewer_analysis.analysis.utils import categorize_flow, calculate_diurnal, decompose_flow, plot_diurnal, plot_decomposition
from sewer_analysis.apps.storm_selector_app import run_storm_selector_app
from sewer_analysis.analysis.RTK_utils import run_RTK

class FlowSite:
    def __init__(self, name: str, catchment_area: float, raw_flow: pd.DataFrame, raw_rainfall: pd.DataFrame, separate_fridays: bool = False):
        self.name = name
        self.catchment_area = catchment_area
        self.runoff_ratio = None
        self.raw_data = RawData(raw_flow, raw_rainfall)
        self.results = ResultsData()
        self.separate_fridays = separate_fridays

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
        self.results.dwf_results = categorize_flow(
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

    def calculate_diurnal(self, plot = True):
        df_input = self.results.dwf_results.copy()
        self.results.diurnal_pattern = calculate_diurnal(df_input)

        if plot == True:
            self._plot_diurnal()

    def _plot_diurnal(self):
        if self.results.diurnal_pattern is None or self.results.dwf_results is None:
            raise RuntimeError("You must run calculate_diurnal() before plotting diurnal pattern")
        plot_diurnal(self.results.diurnal_pattern, self.results.dwf_results)

    def decompose_flow(self, plot = True):
        if self.results.diurnal_pattern is None:
            raise RuntimeError("You must run calculate_diurnal() before calculating RDII")
        
        df_flow = self.raw_data.data.copy()
        df_diurnal = self.results.diurnal_pattern.copy()

        self.results.rdii_results = decompose_flow(df_flow, df_diurnal, self.separate_fridays)

        if plot == True:
            self._plot_decomposition()

    def _plot_decomposition(self):
        plot_decomposition(self.results.rdii_results)

    def _calculate_Ro(self):
        df_rdii = self.results.rdii_results
        Q = df_rdii['RDII']
        P = df_rdii['rainfall_mm']

        # Calculate volume of rain
        V_rain = np.nansum(P) * (1/1000) * self.catchment_area * 10000 * 1000 # Litres

        # Calculate volume of RDII
        V_RDII = np.nansum(Q) * 300 # Litres

        # Calculate runoff ratio
        Ro = V_RDII/V_rain

        return Ro
    
    def select_RTK_storms(self):
        if self.results.rdii_results is None:
            raise RuntimeError("You must have run decompose_flow() before selecting RTK storms")
        run_storm_selector_app(self.results.rdii_results)

    def RTK_method(self, pop_size, max_gens):
        try:
            selected_storm_dates_df = pd.read_csv("../data/selected_storm_dates.csv", parse_dates=['start_date', 'end_date'])
        except Exception:
            raise RuntimeError("No selected storm dates csv found")
        
        Ro = self._calculate_Ro()
        
        run_RTK(self.results.rdii_results, selected_storm_dates_df, self.catchment_area, Ro, pop_size, max_gens)

    def envelope_method(self):
        pass

