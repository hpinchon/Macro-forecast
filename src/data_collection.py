"""
Basic FRED API data collector
Fetches key economic indicators for forecasting
"""
import pandas as pd
import requests
import time
from datetime import datetime
from pathlib import Path

class FREDCollector:
    """Simple FRED API data collector"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        
    def get_series(self, series_id, start_date='2000-01-01'):
        """
        Fetch a single economic time series
        
        Args:
            series_id (str): FRED series identifier (e.g., 'GDPC1')
            start_date (str): Start date in YYYY-MM-DD format
        
        Returns:
            pd.Series: Time series data with date index
        """
        # Build the API request URL
        url = f"{self.base_url}/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date
        }
        
        try:
            # Make the API request
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises exception for bad status codes
            
            # Parse JSON response
            data = response.json()
            observations = data['observations']
            
            # Convert to pandas Series
            dates = [obs['date'] for obs in observations]
            values = [float(obs['value']) if obs['value'] != '.' else None 
                     for obs in observations]
            
            series = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
            series = series.dropna()  # Remove missing values
            
            print(f"✓ Successfully fetched {series_id}: {len(series)} observations")
            return series
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching {series_id}: {e}")
            return pd.Series(dtype=float, name=series_id)
        except Exception as e:
            print(f"✗ Unexpected error for {series_id}: {e}")
            return pd.Series(dtype=float, name=series_id)
    
    def get_multiple_series(self, series_dict, start_date='2000-01-01'):
        """
        Fetch multiple economic series and combine into DataFrame
        
        Args:
            series_dict (dict): {'name': 'FRED_ID'} mapping
            start_date (str): Start date for all series
            
        Returns:
            pd.DataFrame: Combined economic data
        """
        all_series = {}
        
        for name, series_id in series_dict.items():
            series = self.get_series(series_id, start_date)
            if not series.empty:
                all_series[name] = series
            
            # Be nice to the API - small delay between requests
            time.sleep(0.1)
        
        if all_series:
            df = pd.DataFrame(all_series)
            print(f"\n📊 Combined dataset shape: {df.shape}")
            print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
            return df
        else:
            print("❌ No data retrieved")
            return pd.DataFrame()
    
    def save_data_to_csv(self, data, filename='economic_data.csv'):
        """
        Save the collected data to CSV file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Name of the CSV file
            
        Returns:
            Path: Path to the saved file
        """
        # Create the raw data directory if it doesn't exist
        raw_data_path = Path('data/raw')
        raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        filepath = raw_data_path / filename
        data.to_csv(filepath)
        
        print("💾 Data saved to: {}".format(filepath))
        print("📊 File size: {:.1f} KB".format(filepath.stat().st_size / 1024))
        print("📈 Data shape: {}".format(data.shape))
        
        return filepath
    
    def inspect_data(self, data):
        """
        Display comprehensive data inspection
        
        Args:
            data (pd.DataFrame): Data to inspect
        """
        print("\n" + "="*60)
        print("📊 DATA INSPECTION REPORT")
        print("="*60)
        
        # Basic info
        print(f"📏 Shape: {data.shape}")
        print(f"📋 Columns: {list(data.columns)}")
        print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
        print(f"🔢 Data types:\n{data.dtypes}")
        
        # Missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            print(f"\n❓ Missing values:")
            for col, count in missing[missing > 0].items():
                pct = (count / len(data)) * 100
                print(f"   {col}: {count} ({pct:.1f}%)")
        else:
            print(f"\n✅ No missing values")
        
        # First and last few rows
        print(f"\n📈 First 5 rows:")
        print(data.head())
        
        print(f"\n📈 Last 5 rows:")
        print(data.tail())
        
        # Basic statistics
        print(f"\n📊 Basic statistics:")
        print(data.describe())

# Test the collector with the new save functionality
if __name__ == "__main__":
    # Key economic indicators for macro research
    ECONOMIC_SERIES = {
        'GDP': 'GDPC1',           # Real GDP (Quarterly)
        'UNEMPLOYMENT': 'UNRATE', # Unemployment Rate (Monthly)
        'INFLATION': 'CPIAUCSL',  # Consumer Price Index (Monthly)
        'FED_RATE': 'FEDFUNDS',   # Federal Funds Rate (Monthly)
        'EMPLOYMENT': 'PAYEMS'    # Non-farm Payrolls (Monthly)
    }
    
    # Replace with your actual FRED API key
    API_KEY = "073a7aa47f4414e0e5d59ec7119f83ff"
    
    collector = FREDCollector(API_KEY)
    
    # Collect the data
    print("🚀 Starting data collection...")
    economic_data = collector.get_multiple_series(ECONOMIC_SERIES, start_date='2010-01-01')
    
    if not economic_data.empty:
        # Inspect the data
        collector.inspect_data(economic_data)
        
        # Save the data
        saved_file = collector.save_data_to_csv(economic_data, 'economic_data.csv')
        
        print(f"\n✅ SUCCESS! Check your data at: {saved_file}")
        print("📁 You can now find your raw data in the data/raw/ folder")
    else:
        print("❌ No data was collected. Check your API key and internet connection.")
