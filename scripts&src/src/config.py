"""
Configuration file for economic forecasting project
Contains paths, API keys, and model parameters
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
FRED_API_KEY = os.getenv('FRED_API_KEY', 'your_fred_api_key_here')

# Economic indicators to fetch
ECONOMIC_INDICATORS = {
    'GDP': 'GDPC1',           # Real GDP
    'UNEMPLOYMENT': 'UNRATE', # Unemployment Rate
    'INFLATION': 'CPIAUCSL',  # Consumer Price Index
    'FED_RATE': 'FEDFUNDS',   # Federal Funds Rate
    'EMPLOYMENT': 'PAYEMS'    # Non-farm Payrolls
}

# Model parameters
MODEL_CONFIG = {
    'train_test_split': 0.8,
    'arima_order': (2, 1, 1),
    'forecast_periods': 12,
    'confidence_interval': 0.95
}

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (12, 8)
