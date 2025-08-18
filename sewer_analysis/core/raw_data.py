import pandas as pd

class RawData:
    def __init__(self, flow_data: pd.DataFrame, rainfall_data: pd.DataFrame):
        self._validate_dataframe(
            flow_data,
            required_columns={"timestamp", "flow_lps"},
            column_types={"timestamp": "datetime64", "flow_lps": "float"},
            name="flow_data"
        )
        self._validate_dataframe(
            rainfall_data,
            required_columns={"timestamp", "rainfall_mm"},
            column_types={"timestamp": "datetime64", "rainfall_mm": "float"},
            name="rainfall_data"
        )

        self.flow_data = flow_data
        self.rainfall_data = rainfall_data
        self.data = self._combine_flow_rainfall()

    def _validate_dataframe(self, df, required_columns, column_types, name):
        # Check that inputs are pandas dataframes
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
        
        # Check that required columns exist
        if not required_columns.issubset(df.columns):
            raise ValueError(f"{name} must contain columns: {required_columns}")
        
        # Check that dataframe is not empty
        if df.empty:
            raise ValueError(f"{name} must not be empty.")
        
        # Check that column datatypes are correct
        for col, dtype in column_types.items():
            if dtype == "datetime64":
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    raise TypeError(f"{name} '{col}' must be of datetime type.")
            elif dtype == "float":
                if not pd.api.types.is_float_dtype(df[col]):
                    raise TypeError(f"{name} '{col}' must be of float (double) type.")

    def _combine_flow_rainfall(self, time_interval = "5min"):
        # Get min and max timestamps from flow data
        min_flow_time = self.flow_data['timestamp'].min()
        max_flow_time = self.flow_data['timestamp'].max()

        # Get min and max timestamps from rainfall data
        min_rainfall_time = self.rainfall_data['timestamp'].min()
        max_rainfall_time = self.rainfall_data['timestamp'].max()

        if min_rainfall_time > min_flow_time or max_rainfall_time < max_flow_time:
            raise ValueError("Rainfall data not available for entire flow data period")

        # Create a new dataframe with a timestamp column at specified intervals
        timestamps = pd.date_range(start = min_flow_time, end = max_flow_time, freq = time_interval)
        df_combined = pd.DataFrame({'timestamp': timestamps})

        # Merge flow and rainfall data
        df_combined = pd.merge(df_combined, self.flow_data[['timestamp', 'flow_lps']], on='timestamp', how='left')
        df_combined = pd.merge(df_combined, self.rainfall_data[['timestamp', 'rainfall_mm']], on='timestamp', how='left')
        
        return df_combined