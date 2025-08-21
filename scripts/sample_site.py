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

client = FlowWorksClient("ahoover", "Tristan2295!")

# site_name = "STC_SAN024"
# site_area_ha = 47.526
# site_name = "SOH_424879_US2"
# site_area_ha = 52.492
site_name = "CHE_406956_US2"
site_area_ha = 75.578

# Download data for SOH_408121_US2 and McCleery Rain Gauge
raw_flow = client.dl_channel(
        site_name, "Final Flow"
    ).rename(columns={"Timestamp": "timestamp", "Final Flow": "flow_lps"})
raw_rainfall = client.dl_channel(
        "Eric Hamber Secondary Rain Gauge",
        "FINAL Rainfall",
        start_date = "2023-09-01T00:00:00", 
        end_date = "2024-06-01T00:00:00"
    ).rename(columns={"Timestamp": "timestamp", "FINAL Rainfall": "rainfall_mm"})

# %%
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
    plot = False
)

# %%
# Calculate diurnal pattern from dry categorized flow
flow_site.calculate_diurnal(smooth_window = 7, plot = True)

# %%
# Decompose raw input flow series into its components RDII, GWI, and SF
flow_site.decompose_flow(plot = False)

# %%
# Select storms to fit RTK method to. Storms are saved in data/selected_storm_dates.csv.
flow_site.select_RTK_storms()

# %%
# Run genetic algorithm to solve for R, T, K values. Save results in a variable.
# res = flow_site.RTK_method(250, 100)

# %%
#Optional, save the results to a pickle file for later use / archiving
import pickle
# with open('sample_res_CHE_406956_US2.pkl', 'wb') as f:
#     pickle.dump(res, f)

with open('sample_res_CHE_406956_US2.pkl', 'rb') as f:
    res = pickle.load(f)

# %%
flow_site.print_RTK_values()
# %%
flow_site.plot_synthetic_hydrograph()
# %%
flow_site.plot_simulated_flow()
# %%
