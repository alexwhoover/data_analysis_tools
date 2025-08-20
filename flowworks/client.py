from .utils import merge_dataframes
import requests
import pandas as pd
import time

class FlowWorksClient:
    def __init__(self, username, password, base_url = "https://developers.flowworks.com/fwapi/v2"):
        self.username = username
        self.password = password
        self.base_url = base_url.rstrip('/')
        self._token = self._authenticate()
        self.site_lookup_df = self._get_site_lookup()

    # ------------------------------------
    # AUTHENTICATION ####
    # ------------------------------------

    def _authenticate(self) -> str:
        """
        Requests bearer token from FlowWorks based on username and password
        Returns: authentication token (str)
        Raises: ValueError if response code from FlowWorks is 401, meaning username and password are wrong
        """

        url = f"{self.base_url}/tokens"
        resp = requests.post(url, json = {
            "UserName": self.username,
            "Password": self.password
        })
        if resp.status_code == 401:
            raise ValueError("Unauthorized: Check your credentials.")
        
        resp.raise_for_status()
        return resp.json()
    
    def _headers(self) -> dict:
        """
        Helper function to create authentication header for http requests to FlowWorks
        Returns: Dictionary in format 'Authorization: Bearer {token}' (dict)
        """
        if not self._token:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        return {"Authorization": f"Bearer {self._token}"}
    
    # ------------------------------------
    # SITE FUNCTIONS ####
    # ------------------------------------

    def _get_site_lookup(self) -> pd.DataFrame:
        """
        Return DataFrame of all sites. Caches the result as an attribute. (pd.DataFrame)
        """
        url = f"{self.base_url}/sites"
        resp = requests.get(url, headers=self._headers())
        resp.raise_for_status()
        resources = resp.json().get("Resources", [])
        site_lookup_df = pd.DataFrame([
            {"site_name": r["Name"], "site_id": r["Id"]} for r in resources
        ])
        return site_lookup_df
    
    def _get_site_id_from_name(self, site_name) -> str:
        """
        Return the exact matching site_id for a site_name. (str)
        """
        df = self.site_lookup_df
        exact = df[df["site_name"] == site_name]
        if exact.empty:
            raise ValueError(f"No site found with name '{site_name}'.")
        if len(exact) > 1:
            print("Warning: Multiple sites found with same name. Returning ID of first entry.")
        return exact.iloc[0]["site_id"]
    
    # ------------------------------------
    # CHANNEL FUNCTIONS ####
    # ------------------------------------
    
    def _get_channels(self, site_id) -> pd.DataFrame:
        """
        Return DataFrame of all channels for a site.
        """
        url = f"{self.base_url}/sites/{site_id}/channels"
        resp = requests.get(url, headers=self._headers())
        resp.raise_for_status()
        resources = resp.json().get("Resources", [])
        df =  pd.DataFrame([{
            "site_id": site_id,
            "channel_name": r["Name"],
            "channel_id": r["Id"],
            "channel_type": r.get("ChannelType"),
            "channel_unit": r.get("Unit")
        } for r in resources])

        return df
    
    def _get_channel_id_from_name(self, site_id, channel_name) -> str:
        """
        Return channel id corresponding to site_id, channel_name
        """
        channels_df = self._get_channels(site_id)
        match = channels_df[channels_df["channel_name"] == channel_name]
        if match.empty:
            raise ValueError(f"Channel '{channel_name}' does not exist for site {site_id}")
        return match.iloc[0]["channel_id"]
    
    # ------------------------------------
    # DOWNLOAD FUNCTIONS ####
    # ------------------------------------

    def _dl_channel_by_id(self, site_id, channel_id, value_header = "Value", start_date = None, end_date = None, interval_type = None, interval_number = None) -> pd.DataFrame:
        """
        Download data for a given channel ID.
        Returns dataframe with column headers 'Timestamp' and '{value_header}'
        """
        url = f"{self.base_url}/sites/{site_id}/channels/{channel_id}/data"
        params = {
            "startDateFilter": start_date,
            "endDateFilter": end_date,
            "intervalTypeFilter": interval_type,
            "intervalNumberFilter": interval_number
        }

        resp = requests.get(url, headers = self._headers(), params = params)

        if resp.status_code == 401:
            raise ValueError("Unauthorized: Check inputs")
        
        resp.raise_for_status()
        resources = resp.json().get("Resources", [])

        if not resources:
            raise ValueError("Request successful but no data returned.")

        df = pd.DataFrame([{
            "Timestamp": pd.to_datetime(r["DataTime"], format="%Y-%m-%dT%H:%M:%S"),
            value_header: r["DataValue"]
        } for r in resources])

        # Convert the value column to float, coercing errors
        df[value_header] = pd.to_numeric(df[value_header], errors='coerce')
        return df
    
    def dl_channel(self, site_name, channel_name, **kwargs):
        site_id = self._get_site_id_from_name(site_name)
        channel_id = self._get_channel_id_from_name(site_id, channel_name)
        
        df = self._dl_channel_by_id(site_id, channel_id, channel_name, **kwargs)
        return df
    
    def dl_channels(self, site_name, channel_list, **kwargs):
        dfs = []
        for channel_name in channel_list:
            df = self.dl_channel(site_name, channel_name, **kwargs)
            # Rename value column to channel name for clarity
            df = df.rename(columns={"value": channel_name})
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        
        df_merged = merge_dataframes(dfs, on = "Timestamp", how = "outer")

        return df_merged
    
    def dl_channel_to_csv(self, site_name, channel_name, path, **kwargs):
        start_time = time.time()

        df = self.dl_channels(site_name, channel_name, **kwargs)
        header_df = pd.DataFrame([[site_name]])  # one-row header
        header_df.to_csv(path, index=False, header=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Downloaded channel {channel_name} for site {site_name} from FlowWorks in {elapsed_time:.2f} seconds")

        start_time = time.time()

        df.to_csv(path, index=False, mode='a')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Exported to CSV in {elapsed_time:.2f} seconds")

    def dl_channels_to_csv(self, site_name, channel_list, path, **kwargs):
        start_time = time.time()

        df = self.dl_channels(site_name, channel_list, **kwargs)
        header_df = pd.DataFrame([[site_name]])  # one-row header
        header_df.to_csv(path, index=False, header=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Downloaded channel data for site {site_name} from FlowWorks in {elapsed_time:.2f} seconds")

        start_time = time.time()

        df.to_csv(path, index=False, mode='a')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Exported to CSV in {elapsed_time:.2f} seconds")

