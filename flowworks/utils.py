import pandas as pd

def merge_dataframes(dfs, on = "Timestamp", how = "outer") -> pd.DataFrame:
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = pd.merge(df_merged, df, on = on, how = how)

    df_merged = df_merged.sort_values(on).reset_index(drop = True)
    return df_merged