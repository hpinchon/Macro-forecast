"""
Economic data preprocessing for time series forecasting
Handles stationarity, missing values, and feature engineering
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EconomicPreprocessor:
    """Comprehensive preprocessing for economic time series"""
    
    def __init__(self):
        self.transformation_log = {}
        self.original_data = None
        
    def load_data(self, filepath='data/raw/economic_data.csv'):
        """
        Load economic data from CSV file
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded economic data
        """
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.original_data = df.copy()
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"📋 Columns: {list(df.columns)}")
        
        return df
    
    def check_stationarity(self, series, name="Series", alpha=0.05):
        """
        Test for stationarity using ADF test
        
        Args:
            series (pd.Series): Time series to test
            name (str): Name for reporting
            alpha (float): Significance level
            
        Returns:
            dict: Test results
        """
        # Remove NaN values for testing
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return {
                'series_name': name,
                'is_stationary': False,
                'reason': 'Insufficient data points',
                'adf_pvalue': None
            }
        
        # Augmented Dickey-Fuller test
        # H0: Series has unit root (non-stationary)
        # H1: Series is stationary
        adf_result = adfuller(clean_series, autolag='AIC')
        
        is_stationary = adf_result[1] < alpha
        
        result = {
            'series_name': name,
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': is_stationary,
            'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
        }
        
        print(f"\n🔍 Stationarity Test: {name}")
        print(f"   ADF Statistic: {adf_result[0]:.4f}")
        print(f"   p-value: {adf_result[1]:.4f}")
        print(f"   Critical Values:")
        for key, value in adf_result[4].items():
            print(f"      {key}: {value:.4f}")
        print(f"   Result: {result['interpretation']}")
        
        return result
    
    def make_stationary(self, series, name="Series", max_diff=2):
        """
        Transform series to achieve stationarity
        
        Args:
            series (pd.Series): Input time series
            name (str): Series name for logging
            max_diff (int): Maximum differencing orders to try
            
        Returns:
            tuple: (transformed_series, transformation_info)
        """
        print(f"\n🔧 Making {name} stationary...")
        
        # Test original series first
        original_test = self.check_stationarity(series, f"{name} (Original)")
        if original_test['is_stationary']:
            print(f"✅ {name} is already stationary!")
            return series, {'method': 'none', 'parameters': {}}
        
        # Try log transformation for positive series with exponential growth
        if (series > 0).all() and series.std() / series.mean() > 0.1:
            print(f"🧮 Trying log transformation for {name}...")
            log_series = np.log(series)
            log_test = self.check_stationarity(log_series, f"{name} (Log)")
            
            if log_test['is_stationary']:
                print(f"✅ {name}: Log transformation successful!")
                return log_series, {'method': 'log', 'parameters': {}}
        
        # Try differencing
        current_series = series.copy()
        for diff_order in range(1, max_diff + 1):
            print(f"🔄 Trying differencing order {diff_order} for {name}...")
            diff_series = current_series.diff().dropna()
            diff_test = self.check_stationarity(diff_series, f"{name} (Diff {diff_order})")
            
            if diff_test['is_stationary']:
                print(f"✅ {name}: Differencing order {diff_order} successful!")
                return diff_series, {'method': 'difference', 'parameters': {'order': diff_order}}
            
            current_series = diff_series
        
        # Try log + differencing for stubborn series
        if (series > 0).all():
            print(f"🧮 Trying log + differencing for {name}...")
            log_series = np.log(series)
            log_diff = log_series.diff().dropna()
            log_diff_test = self.check_stationarity(log_diff, f"{name} (Log+Diff)")
            
            if log_diff_test['is_stationary']:
                print(f"✅ {name}: Log + differencing successful!")
                return log_diff, {'method': 'log_difference', 'parameters': {'order': 1}}
        
        # If nothing works, return first difference
        print(f"⚠️ {name}: Using first difference as fallback")
        return series.diff().dropna(), {'method': 'difference_fallback', 'parameters': {'order': 1}}
    
    def handle_missing_values(self, df, method='interpolate'):
        """
        Handle missing values using appropriate time series methods
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Method to handle missing values
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        print(f"\n🔧 Handling Missing Values (Method: {method})")
        
        # Report missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            print(f"📊 Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                pct = (count / len(df)) * 100
                print(f"   {col}: {count} ({pct:.1f}%)")
        else:
            print("✅ No missing values found!")
            return df
        
        df_clean = df.copy()
        
        if method == 'interpolate':
            # Linear interpolation - best for time series
            df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            # Forward fill - assumes values persist
            df_clean = df_clean.fillna(method='ffill')
        elif method == 'backward_fill':
            # Backward fill
            df_clean = df_clean.fillna(method='bfill')
        elif method == 'drop':
            # Drop rows with any missing values
            df_clean = df_clean.dropna()
        
        # Final check
        remaining_missing = df_clean.isnull().sum().sum()
        print(f"✅ Missing values after cleaning: {remaining_missing}")
        print(f"📊 Final shape: {df_clean.shape}")
        
        return df_clean
    
    def create_features(self, df, lags=[1, 3, 6, 12], windows=[3, 6, 12]):
        """
        Create lagged and rolling window features
        
        Args:
            df (pd.DataFrame): Input dataframe
            lags (list): Lag periods to create
            windows (list): Rolling window sizes
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        print(f"\n🔄 Creating Features...")
        print(f"   Lags: {lags}")
        print(f"   Rolling windows: {windows}")
        
        df_features = df.copy()
        original_cols = df.columns.tolist()
        
        # Create lagged features
        for col in original_cols:
            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                df_features[lag_col] = df[col].shift(lag)
        
        # Create rolling features
        for col in original_cols:
            for window in windows:
                # Moving average
                ma_col = f"{col}_ma_{window}"
                df_features[ma_col] = df[col].rolling(window=window).mean()
                
                # Rolling standard deviation (volatility)
                std_col = f"{col}_std_{window}"
                df_features[std_col] = df[col].rolling(window=window).std()
        
        print(f"✅ Features created!")
        print(f"   Original features: {len(original_cols)}")
        print(f"   Total features: {df_features.shape[1]}")
        print(f"   Added features: {df_features.shape[1] - len(original_cols)}")
        
        return df_features
    
    def preprocess_pipeline(self, filepath='data/raw/economic_data.csv', save_results=True):
        """
        Complete preprocessing pipeline
        
        Args:
            filepath (str): Path to raw data
            save_results (bool): Whether to save processed data
            
        Returns:
            tuple: (processed_dataframe, metadata)
        """
        print("🚀 Starting Economic Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        df = self.load_data(filepath)
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values(df, method='interpolate')
        
        # Step 3: Make series stationary
        print(f"\n📈 STATIONARITY ANALYSIS")
        print("=" * 40)
        
        df_stationary = pd.DataFrame(index=df_clean.index)
        
        for column in df_clean.columns:
            series_stationary, transform_info = self.make_stationary(
                df_clean[column], name=column
            )
            df_stationary[column] = series_stationary
            self.transformation_log[column] = transform_info
        
        # Step 4: Remove NaN values created by transformations
        df_stationary = df_stationary.dropna()
        
        # Step 5: Create additional features
        print(f"\n🔧 FEATURE ENGINEERING")
        print("=" * 40)
        df_final = self.create_features(df_stationary)
        
        # Step 6: Final cleanup
        df_final = df_final.dropna()
        
        # Prepare metadata
        metadata = {
            'original_shape': df.shape,
            'stationary_shape': df_stationary.shape,
            'final_shape': df_final.shape,
            'date_range': (df_final.index.min(), df_final.index.max()),
            'transformations': self.transformation_log,
            'features_created': df_final.shape[1] - df.shape[1],
            'data_loss_pct': ((len(df) - len(df_final)) / len(df)) * 100
        }
        
        # Save results
        if save_results:
            output_dir = Path('data/processed')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save processed data
            output_file = output_dir / 'processed_economic_data.csv'
            df_final.to_csv(output_file)
            
            # Save metadata
            metadata_file = output_dir / 'preprocessing_metadata.txt'
            with open(metadata_file, 'w') as f:
                f.write("PREPROCESSING METADATA\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Original shape: {metadata['original_shape']}\n")
                f.write(f"Final shape: {metadata['final_shape']}\n")
                f.write(f"Data loss: {metadata['data_loss_pct']:.1f}%\n\n")
                f.write("TRANSFORMATIONS APPLIED:\n")
                f.write("-" * 30 + "\n")
                for series, transform in metadata['transformations'].items():
                    f.write(f"{series}: {transform['method']}\n")
            
            print(f"\n💾 Results saved:")
            print(f"   Data: {output_file}")
            print(f"   Metadata: {metadata_file}")
        
        # Final summary
        print(f"\n✅ PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"📊 Original shape: {metadata['original_shape']}")
        print(f"📊 Final shape: {metadata['final_shape']}")
        print(f"📊 Features added: {metadata['features_created']}")
        print(f"📊 Data loss: {metadata['data_loss_pct']:.1f}%")
        
        print(f"\n📋 Transformations Applied:")
        for series, transform in metadata['transformations'].items():
            print(f"   {series}: {transform['method']}")
        
        return df_final, metadata

# Test the preprocessor
if __name__ == "__main__":
    preprocessor = EconomicPreprocessor()
    processed_data, metadata = preprocessor.preprocess_pipeline()
    
    print(f"\n🎯 Ready for modeling!")
    print(f"Processed data shape: {processed_data.shape}")
