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
###################################
# DOWNLOAD DATA FROM FLOWWORKS ####
###################################

client = FlowWorksClient("ahoover", "Tristan2295!")

site_name = "SOH_408121_US2"

# Download data for SOH_408121_US2 and McCleery Rain Gauge
raw_flow = client.dl_channel(
        site_name, "Final Flow"
    ).rename(columns={"Timestamp": "timestamp", "Final Flow": "flow_lps"})
raw_rainfall = client.dl_channel(
        "McCleery Golf Course Rain Gauge",
        "FINAL Rainfall", 
        start_date = "2024-09-01T00:00:00", 
        end_date = "2025-06-01T00:00:00"
    ).rename(columns={"Timestamp": "timestamp", "FINAL Rainfall": "rainfall_mm"})

# %%
###################################
# ANALYSIS  #######################
###################################

# Create a FlowSite object
flow_site = FlowSite(site_name, raw_flow, raw_rainfall, separate_fridays = False)

# Categorize Flow
flow_site.categorize_flow(            
    rolling_window_hr = 6,
    min_intensity_mm_hr = 0.1,
    inter_event_duration_hr = 24,
    min_event_duration_hr = 1,
    response_time_hr = 72,
    lead_time_hr = 2,
    plot = False
)

flow_site.calculate_diurnal(plot = False)
flow_site.decompose_flow(plot = True)


