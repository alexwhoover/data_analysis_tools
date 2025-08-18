import pandas as pd
from sewer_analysis.core.raw_data import RawData
from sewer_analysis.core.results_data import ResultsData
from sewer_analysis.analysis.utils import categorize_flow, plot_categorization, calculate_diurnal, plot_diurnal

class FlowSite:
    def __init__(self, name: str, raw_data: RawData, separate_fridays: bool = False):
        self.name = name
        self.raw_data = raw_data
        self.results = ResultsData()
        self.separate_fridays = separate_fridays
        self._selected_dwf_days = pd.Series()

    def categorize_flow(
            self,
            rolling_window_hr = 6,
            min_intensity_mm_hr = 0.1,
            inter_event_duration_hr = 24,
            min_event_duration_hr = 1,
            response_time_hr = 72,
            lead_time_hr = 2
    ):
        df_input = self.raw_data.data.copy()
        self.results.categorization_results = categorize_flow(
            df_input, 
            rolling_window_hr = rolling_window_hr, 
            min_intensity_mm_hr = min_intensity_mm_hr, 
            inter_event_duration_hr = inter_event_duration_hr,
            min_event_duration_hr = min_event_duration_hr,
            response_time_hr = response_time_hr,
            lead_time_hr = lead_time_hr
        )

    def plot_categorization(self):
        if self.results.categorization_results is None:
            raise RuntimeError("You must run categorize_flow() before plotting categorization.")
        plot_categorization(self.results.categorization_results)

    def calculate_diurnal(self):
        df_input = self.results.categorization_results.copy()
        self.results.diurnal_pattern, self.results.dwf_results = calculate_diurnal(df_input, self.separate_fridays)

    def plot_diurnal(self):
        if self.results.diurnal_pattern is None or self.results.dwf_results is None:
            raise RuntimeError("You must run calculate_diurnal() before plotting diurnal pattern")
        plot_diurnal(self.results.diurnal_pattern, self.results.dwf_results)

    def calculate_RDII(self):
        pass