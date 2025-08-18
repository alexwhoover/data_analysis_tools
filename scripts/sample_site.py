# %%
# Set Project Root
import sys
sys.path.append(r"c:\Git\DataAnalysisTools")

# Import Libraries
import pandas as pd

from sewer_analysis.core.flow_site import FlowSite
from sewer_analysis.core.raw_data import RawData
from flowworks.client import FlowWorksClient

# %%
client = FlowWorksClient("ahoover", "Tristan2295!")

# Download data for STC_SAN024 and McCleery Rain Gauge
raw_flow = client.dl_channel(
        "STC_SAN024", "Final Flow"
    ).rename(columns={"Timestamp": "timestamp", "Final Flow": "flow_lps"})
raw_rainfall = client.dl_channel(
        "McCleery Golf Course Rain Gauge",
        "FINAL Rainfall", 
        start_date = "2023-09-01T00:00:00", 
        end_date = "2025-05-01T00:00:00"
    ).rename(columns={"Timestamp": "timestamp", "FINAL Rainfall": "rainfall_mm"})

# Convert value dtypes from object to float
raw_flow['flow_lps'] = pd.to_numeric(raw_flow['flow_lps'], errors='coerce')
raw_rainfall['rainfall_mm'] = pd.to_numeric(raw_rainfall['rainfall_mm'], errors='coerce')

# %%
# Create a SiteData object
raw_data = RawData(raw_flow, raw_rainfall)

# Create a FlowSite object
flow_site = FlowSite("STC_SAN024", raw_data)

# %%
# Categorize Flow
flow_site.categorize_flow(            
    rolling_window_hr = 6,
    min_intensity_mm_hr = 0.1,
    inter_event_duration_hr = 24,
    min_event_duration_hr = 1,
    response_time_hr = 72,
    lead_time_hr = 2
)

flow_site.plot_categorization()

# %%
flow_site.calculate_diurnal()
flow_site.plot_diurnal()
# %%
