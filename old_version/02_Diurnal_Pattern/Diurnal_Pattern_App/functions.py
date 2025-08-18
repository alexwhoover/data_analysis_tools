"""
Purpose: Calculate base flow from a diurnal pattern using the Stevens - Schutzback Method
Inputs:
    MDF (float): Minimum Daily Flow Rate (L/s) (I am calling this MNF)
    ADF (float): Average Daily Flow Rate (L/s) (I am calling this ADWF)

Returns:
    BI (float): Base Infiltration (L/s) (I am calling this GWI)

Source: 
    Mitchell, Paul & Stevens, Patrick & Nazaroff, Adam. (2007). QUANTIFYING BASE INFILTRATION IN SEWERS: A Comparison of Methods and a Simple Empirical Solution. Proceedings of the Water Environment Federation. 2007. 219-238. 10.2175/193864707787974805. 

"""
def calc_base_flow(MDF: float, ADF: float):
    conversion_factor = 0.022824465227271 # 0.023mgd = 1L/s

    # Convert lps to mgd
    MDF_mgd = MDF * conversion_factor
    ADF_mgd = ADF * conversion_factor

    # Stevens-Shutzbach Method Equation
    BI_mgd = (0.4 * MDF_mgd) / (1 - 0.6 * ((MDF_mgd / ADF_mgd)**(ADF_mgd**0.7)))
    
    # Convert mgd to lps
    BI = BI_mgd / conversion_factor

    return BI

"""
Purpose: To filter the input dataframe resulting from flow_categorization.csv to only include dry full days
Inputs:
- df (pd.DataFrame): Dataframe resulting from Flow_Categorization.py script output
Outupts:
- df (pd.DataFrame): Filtered dataframe containing only fully dry days, with added columns date, time_of_day, and weekday
"""
def preprocess_data(df):
    df['']