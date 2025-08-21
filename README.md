# Sewer Data Analysis Tools
### Author: Alex Hoover, EIT (alexwhoover@gmail.com or alex.hoover@vancouver.ca)| Last Updated: 2025-08-21

A Python package for analyzing sewer flow data to quantify infiltration and inflow (I&I) using the FlowSite class and RTK unit hydrograph modeling.

## Overview

This package processes raw 5-minute sewer flow and rainfall data to categorize flow periods, calculate diurnal patterns, and calibrate RTK parameters using a genetic algorithm. Class methods are written assuming 5-minute time interval data, but could be generalized in future.

## Example Usage

```python
from sewer_analysis.core.flow_site import FlowSite

# ... Import Flow and Rainfall Data ...

# Create FlowSite object
flow_site = FlowSite("site_name", catchment_area_ha, raw_flow, raw_rainfall)

# Analysis workflow

# Categorize flow into wet and dry periods
flow_site.categorize_flow(          
    rolling_window_hr = 6,
    min_intensity_mm_hr = 0.1,
    inter_event_duration_hr = 24,
    min_event_duration_hr = 1,
    response_time_hr = 72,
    lead_time_hr = 2,
    plot = False
)

# Calculate diurnal patterns
flow_site.calculate_diurnal(smooth_window=7, plot=True)

# Decompose flow into RDII, SF, and GWI components
flow_site.decompose_flow(plot=True)

# Select storms for RTK calibration
flow_site.select_RTK_storms()

# Calibrate RTK parameters
flow_site.RTK_method(pop_size=250, max_gens=100)
```

## FlowSite Methods

### `categorize_flow()`
Identifies wet weather events using rolling rainfall intensity:
- Calculates rolling sum over specified window (default: 6 hours)
- Flags periods exceeding intensity threshold (default: 0.1 mm/hr)
- Merges events within inter-event duration (default: 24 hours)
- Applies lead/lag times for flow response (default: 2hr lead, 72hr response)

### `calculate_diurnal()`
Computes typical daily flow patterns:
- Filters to fully dry days (no wet weather, no missing data, 288 timesteps/day)
- Removes outlier days based on daily maximum flow (IQR method)
- Groups by day type (workday vs weekend/holiday)
- Calculates median flow by time of day with optional smoothing
- Estimates groundwater infiltration using Stevens-Schutzback method

### `decompose_flow()`
Separates flow components:
- **RDII**: Observed flow minus diurnal pattern baseline
- **GWI**: Groundwater infiltration from diurnal minimum (MNF) and average (ADWF)
- **SF**: Sanitary flow = Diurnal pattern - GWI

### `select_RTK_storms()`
Interactive storm selection using Dash app:
- Visualizes RDII time series with selectable periods
- Saves selected storm dates to CSV file
- Returns control to main script after user selection

### `RTK_method()`
Genetic algorithm optimization for RTK parameters:
- **Variables**: 9 parameters (R1,T1,K1, R2,T2,K2, R3,T3,K3)
- **Constraints**: 
  - R1, R2, R3 ∈ [0,1], 
  - T1 ∈ [0.1,2], T2 ∈ [2,12], T3 ∈ [12,72]
  - K1 ∈ [2,5], K2 ∈ [1,3], K3 ∈ [0.5,2]
- **Objective**: Maximize Kling-Gupta Efficiency (KGE)
- **Algorithm**: Pymoo GA with SBX crossover and polynomial mutation

### `print_RTK_values()`
### `plot_synthetic_hydrograph()`
### `plot_simulated_flow()`

## RTK Unit Hydrograph Model

Each RTK triplet generates a triangular unit hydrograph:
- **R**: Runoff coefficient (fraction of rainfall contributing to peak)
- **T**: Time to peak (hours)
- **K**: Recession ratio (controls tail length)

Total response = sum of three unit hydrographs convolved with effective rainfall.

## Data Requirements

- **Flow**: 5-minute interval timestamps and flow rates (L/s)
- **Rainfall**: 5-minute interval timestamps and depths (mm)
- Synchronized time periods with minimal missing data

## Key Algorithms

### Stevens-Schutzback GWI Formula
```
GWI = (0.4 * MNF) / (1 - 0.6 * (MNF/ADWF)^(ADWF^0.7))
```

### Runoff Ratio Calculation
```
Ro = RDII_volume / (rainfall_depth * catchment_area)
```

### KGE Objective Function
```
KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
```
Where r = correlation, α = variability ratio, β = bias ratio

## Dependencies

- pandas, numpy: Data processing
- plotly: Visualization  
- pymoo: Genetic algorithm optimization
- dash: Interactive storm selection
- holidays: Day type classification

## Installation

```bash
git clone https://github.com/your-repo/DataAnalysisTools.git
cd DataAnalysisTools
conda env create -f environment.yml
conda activate sewer-analysis
```