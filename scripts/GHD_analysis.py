# %%
# Set Project Root
import sys
sys.path.append(r"c:\Git\DataAnalysisTools")

# Import Libraries
import pandas as pd

from sewer_analysis.core.flow_site import FlowSite
from flowworks.client import FlowWorksClient

# %%
###################################
# DOWNLOAD DATA FROM FLOWWORKS ####
###################################

client = FlowWorksClient("*****", "*****")

site_name = "CHA_SAN004"
site_area_ha = 31.756574

# Download data for SOH_408121_US2 and McCleery Rain Gauge
raw_flow = client.dl_channel(
        site_name, "Final Flow"
    ).rename(columns={"Timestamp": "timestamp", "Final Flow": "flow_lps"})

print(f"Raw flow timestamp range: {raw_flow['timestamp'].min()} to {raw_flow['timestamp'].max()}")

# %%
raw_rainfall = client.dl_channel(
    "Champlain Heights Community Centre Rain Gauge",
    "FINAL Rainfall",
    start_date = raw_flow['timestamp'].min().strftime('%Y-%m-%dT%H:%M:%S'), 
    end_date = raw_flow['timestamp'].max().strftime('%Y-%m-%dT%H:%M:%S')
    ).rename(columns={"Timestamp": "timestamp", "FINAL Rainfall": "rainfall_mm"})

###################################
# ANALYSIS  #######################
###################################

# Create a FlowSite object
flow_site = FlowSite(site_name, site_area_ha, raw_flow, raw_rainfall, separate_fridays = False)

# %%
# Categorize Flow
flow_site.categorize_flow(            
    rolling_window_hr = 6,
    min_intensity_mm_hr = 0.1,
    inter_event_duration_hr = 24,
    min_event_duration_hr = 1,
    response_time_hr = 72,
    lead_time_hr = 2,
    plot = True
)

# %%
# Calculate diurnal pattern from dry categorized flow
flow_site.calculate_diurnal(smooth_window = 7, plot = True)

# %%
# Decompose raw input flow series into its components RDII, GWI, and SF
flow_site.decompose_flow(plot = True)

print(flow_site.results.runoff_ratio)
# %%
