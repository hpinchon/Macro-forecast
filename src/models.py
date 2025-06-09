"""
Advanced Time Series Forecasting Models
Combines traditional econometric and modern ML approaches
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Traditional time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

# Machine learning models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Deep learning models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
    from tensorflow.keras.optimizers import Adam # type: ignore
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️ TensorFlow/Keras not available. LSTM models will be skipped.")

from pathlib import Path
import joblib
from datetime import datetime, timedelta

class EconomicForecaster:
    """Advanced economic forecasting with multiple model types"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.model_dir = Path('models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, filepath='data/processed/processed_economic_data.csv'):
        """Load processed economic data"""
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Focus on main economic indicators for modeling
        main_indicators = [col for col in df.columns if not any(
            suffix in col for suffix in ['_lag_', '_ma_', '_std_']
        )]
        
        print(f"✅ Loaded data for modeling: {df.shape}")
        print(f"📊 Main indicators: {main_indicators}")
        
        return df, main_indicators
    
    def train_test_split(self, data, test_size=0.2):
        """
        Split time series data maintaining temporal order
        
        Args:
            data (pd.DataFrame): Time series data
            test_size (float): Proportion for testing
            
        Returns:
            tuple: (train_data, test_data, split_date)
        """
        split_idx = int(len(data) * (1 - test_size))
        split_date = data.index[split_idx]
        
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        print(f"📊 Data Split:")
        print(f"   Training: {train_data.shape} ({train_data.index.min()} to {train_data.index.max()})")
        print(f"   Testing: {test_data.shape} ({test_data.index.min()} to {test_data.index.max()})")
        
        return train_data, test_data, split_date
    
    def fit_arima_model(self, series, name, order=(1,1,1)):
        """
        Fit ARIMA model for individual economic indicator
        
        Args:
            series (pd.Series): Time series data
            name (str): Series name
            order (tuple): ARIMA order (p,d,q)
            
        Returns:
            fitted model and predictions
        """
        print(f"\n🔧 Fitting ARIMA{order} for {name}...")
        
        try:
            # Fit ARIMA model
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            # Generate in-sample predictions
            predictions = fitted_model.fittedvalues
            
            # Model diagnostics
            ljung_box = acorr_ljungbox(fitted_model.resid, lags=10, return_df=True)
            
            # Store model info
            model_info = {
                'model': fitted_model,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'ljung_box_pvalue': ljung_box['lb_pvalue'].iloc[-1],
                'order': order,
                'predictions': predictions
            }
            
            print(f"✅ ARIMA{order} fitted for {name}")
            print(f"   AIC: {fitted_model.aic:.2f}")
            print(f"   BIC: {fitted_model.bic:.2f}")
            print(f"   Ljung-Box p-value: {ljung_box['lb_pvalue'].iloc[-1]:.4f}")
            
            return model_info
            
        except Exception as e:
            print(f"❌ Error fitting ARIMA for {name}: {str(e)}")
            return None
    
    def auto_arima_selection(self, series, name, max_p=3, max_q=3, max_d=2):
        """
        Automatic ARIMA order selection using AIC
        
        Args:
            series (pd.Series): Time series data
            name (str): Series name
            max_p, max_q, max_d (int): Maximum orders to test
            
        Returns:
            Best ARIMA model info
        """
        print(f"\n🔍 Auto-selecting ARIMA order for {name}...")
        
        best_aic = np.inf
        best_order = None
        best_model_info = None
        
        # Grid search over ARIMA orders
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model_info = self.fit_arima_model(series, f"{name}_temp", (p,d,q))
                        if model_info and model_info['aic'] < best_aic:
                            best_aic = model_info['aic']
                            best_order = (p,d,q)
                            best_model_info = model_info
                    except:
                        continue
        
        if best_model_info:
            print(f"✅ Best ARIMA order for {name}: {best_order} (AIC: {best_aic:.2f})")
            return best_model_info
        else:
            print(f"❌ Could not find suitable ARIMA model for {name}")
            return None
    
    def fit_var_model(self, data, maxlags=4):
        """
        Fit Vector Autoregression (VAR) model for multivariate forecasting
        
        Args:
            data (pd.DataFrame): Multivariate time series
            maxlags (int): Maximum lags to consider
            
        Returns:
            Fitted VAR model info
        """
        print(f"\n🔧 Fitting VAR model with max {maxlags} lags...")
        
        try:
            # Fit VAR model with automatic lag selection
            model = VAR(data)
            fitted_model = model.fit(maxlags=maxlags, ic='aic')
            
            # Model diagnostics
            lag_order = fitted_model.k_ar
            aic = fitted_model.aic
            
            model_info = {
                'model': fitted_model,
                'lag_order': lag_order,
                'aic': aic,
                'predictions': fitted_model.fittedvalues
            }
            
            print(f"✅ VAR model fitted")
            print(f"   Optimal lags: {lag_order}")
            print(f"   AIC: {aic:.2f}")
            
            return model_info
            
        except Exception as e:
            print(f"❌ Error fitting VAR model: {str(e)}")
            return None
    
    def prepare_lstm_data(self, series, lookback=12):
        """
        Prepare data for LSTM model
        
        Args:
            series (pd.Series): Time series data
            lookback (int): Number of past observations to use
            
        Returns:
            tuple: (X, y, scaler)
        """
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    def build_lstm_model(self, input_shape, units=50, dropout=0.2):
        """
        Build LSTM neural network architecture
        
        Args:
            input_shape (tuple): Input shape for LSTM
            units (int): Number of LSTM units
            dropout (float): Dropout rate
            
        Returns:
            Compiled LSTM model
        """
        if not KERAS_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        
        return model
    
    def fit_lstm_model(self, series, name, lookback=12, epochs=50, batch_size=32):
        """
        Fit LSTM model for time series forecasting
        
        Args:
            series (pd.Series): Time series data
            name (str): Series name
            lookback (int): Lookback window
            epochs (int): Training epochs
            batch_size (int): Batch size
            
        Returns:
            LSTM model info
        """
        if not KERAS_AVAILABLE:
            print(f"⚠️ Skipping LSTM for {name} - TensorFlow not available")
            return None
            
        print(f"\n🔧 Fitting LSTM model for {name}...")
        
        try:
            # Prepare data
            X, y, scaler = self.prepare_lstm_data(series, lookback)
            
            if len(X) < 20:  # Need sufficient data
                print(f"⚠️ Insufficient data for LSTM training: {len(X)} samples")
                return None
            
            # Split for training
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build and train model
            model = self.build_lstm_model((X.shape[1], 1))
            
            # Train with validation split
            history = model.fit(X_train, y_train, 
                              epochs=epochs, 
                              batch_size=batch_size,
                              validation_data=(X_test, y_test),
                              verbose=0)
            
            # Generate predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Inverse transform predictions
            train_pred = scaler.inverse_transform(train_pred)
            test_pred = scaler.inverse_transform(test_pred)
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
            
            model_info = {
                'model': model,
                'scaler': scaler,
                'lookback': lookback,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'history': history.history,
                'X_shape': X.shape
            }
            
            print(f"✅ LSTM model fitted for {name}")
            print(f"   Train RMSE: {train_rmse:.4f}")
            print(f"   Test RMSE: {test_rmse:.4f}")
            
            return model_info
            
        except Exception as e:
            print(f"❌ Error fitting LSTM for {name}: {str(e)}")
            return None
    
    def fit_hybrid_model(self, series, name):
        """
        Fit hybrid ARIMA-LSTM model combining linear and nonlinear components
        
        Args:
            series (pd.Series): Time series data
            name (str): Series name
            
        Returns:
            Hybrid model info
        """
        print(f"\n🔧 Fitting Hybrid ARIMA-LSTM for {name}...")
        
        try:
            # Step 1: Fit ARIMA model
            arima_info = self.auto_arima_selection(series, name)
            if not arima_info:
                return None
            
            # Step 2: Extract ARIMA residuals
            arima_residuals = arima_info['model'].resid
            
            # Step 3: Fit LSTM on residuals
            lstm_info = self.fit_lstm_model(arima_residuals, f"{name}_residuals", 
                                          epochs=30, lookback=6)
            
            if not lstm_info:
                print(f"⚠️ LSTM component failed, using ARIMA only for {name}")
                return arima_info
            
            model_info = {
                'arima_component': arima_info,
                'lstm_component': lstm_info,
                'type': 'hybrid_arima_lstm'
            }
            
            print(f"✅ Hybrid ARIMA-LSTM fitted for {name}")
            
            return model_info
            
        except Exception as e:
            print(f"❌ Error fitting hybrid model for {name}: {str(e)}")
            return None
    
    def evaluate_model_performance(self, actual, predicted, model_name):
        """
        Calculate comprehensive model evaluation metrics
        
        Args:
            actual (array): Actual values
            predicted (array): Predicted values
            model_name (str): Model name for reporting
            
        Returns:
            dict: Evaluation metrics
        """
        # Align arrays and remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {'error': 'No valid data points for evaluation'}
        
        # Calculate metrics
        mse = mean_squared_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_clean, predicted_clean)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        
        # Directional accuracy (for economic indicators)
        actual_direction = np.diff(actual_clean) > 0
        pred_direction = np.diff(predicted_clean) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'R_squared': 1 - (np.sum((actual_clean - predicted_clean) ** 2) / 
                             np.sum((actual_clean - np.mean(actual_clean)) ** 2))
        }
        
        print(f"📊 {model_name} Performance:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return metrics
    
    def comprehensive_modeling_pipeline(self, filepath='data/processed/processed_economic_data.csv'):
        """
        Run comprehensive modeling pipeline with multiple approaches
        
        Args:
            filepath (str): Path to processed data
            
        Returns:
            dict: All model results and comparisons
        """
        print("🚀 COMPREHENSIVE ECONOMIC FORECASTING PIPELINE")
        print("=" * 60)
        
        # Load data
        data, main_indicators = self.load_data(filepath)
        main_data = data[main_indicators]
        
        # Train-test split
        train_data, test_data, split_date = self.train_test_split(main_data)
        
        all_results = {}
        
        # 1. Individual ARIMA models for each indicator
        print(f"\n📈 FITTING INDIVIDUAL ARIMA MODELS")
        print("=" * 40)
        
        for indicator in main_indicators:
            train_series = train_data[indicator].dropna()
            test_series = test_data[indicator].dropna()
            
            if len(train_series) < 20:
                print(f"⚠️ Skipping {indicator} - insufficient data")
                continue
            
            # Fit ARIMA
            arima_result = self.auto_arima_selection(train_series, indicator)
            if arima_result:
                # Generate forecasts
                forecast_steps = len(test_series)
                forecast = arima_result['model'].forecast(steps=forecast_steps)
                
                # Evaluate
                metrics = self.evaluate_model_performance(
                    test_series.values, forecast, f"ARIMA_{indicator}"
                )
                
                all_results[f"ARIMA_{indicator}"] = {
                    'model_info': arima_result,
                    'forecast': forecast,
                    'metrics': metrics
                }
        
        # 2. VAR model for multivariate forecasting
        print(f"\n📈 FITTING VAR MODEL")
        print("=" * 25)
        
        var_result = self.fit_var_model(train_data)
        if var_result:
            # Generate VAR forecasts
            forecast_steps = len(test_data)
            var_forecast = var_result['model'].forecast(train_data.values, steps=forecast_steps)
            
            # Evaluate VAR for each variable
            for i, indicator in enumerate(main_indicators):
                if i < var_forecast.shape[1]:
                    test_series = test_data[indicator].dropna()
                    forecast_series = var_forecast[:len(test_series), i]
                    
                    metrics = self.evaluate_model_performance(
                        test_series.values, forecast_series, f"VAR_{indicator}"
                    )
                    
                    all_results[f"VAR_{indicator}"] = {
                        'model_info': var_result,
                        'forecast': forecast_series,
                        'metrics': metrics
                    }
        
        # 3. LSTM models (if available)
        if KERAS_AVAILABLE:
            print(f"\n📈 FITTING LSTM MODELS")
            print("=" * 25)
            
            for indicator in main_indicators:
                full_series = data[indicator].dropna()
                
                if len(full_series) < 50:
                    print(f"⚠️ Skipping LSTM for {indicator} - insufficient data")
                    continue
                
                lstm_result = self.fit_lstm_model(full_series, indicator)
                if lstm_result:
                    all_results[f"LSTM_{indicator}"] = {
                        'model_info': lstm_result,
                        'metrics': {'Train_RMSE': lstm_result['train_rmse'],
                                  'Test_RMSE': lstm_result['test_rmse']}
                    }
        
        # 4. Model comparison and ranking
        print(f"\n📊 MODEL COMPARISON AND RANKING")
        print("=" * 35)
        
        comparison_results = self.compare_model_performance(all_results)
        
        # Save results
        self.save_model_results(all_results, comparison_results)
        
        print(f"\n✅ MODELING PIPELINE COMPLETE!")
        print(f"📁 Results saved to: {self.model_dir}")
        
        return all_results, comparison_results
    
    def compare_model_performance(self, all_results):
        """Compare performance across all models"""
        comparison_data = []
        
        for model_name, result in all_results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                if 'RMSE' in metrics:
                    comparison_data.append({
                        'Model': model_name,
                        'RMSE': metrics['RMSE'],
                        'MAE': metrics.get('MAE', np.nan),
                        'MAPE': metrics.get('MAPE', np.nan),
                        'R_squared': metrics.get('R_squared', np.nan)
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('RMSE')
            
            print("🏆 MODEL RANKINGS (by RMSE):")
            print(comparison_df.to_string(index=False, float_format='%.4f'))
            
            return comparison_df
        
        return pd.DataFrame()
    
    def save_model_results(self, all_results, comparison_results):
        """Save model results and comparisons"""
        # Save comparison results
        comparison_results.to_csv(self.model_dir / 'model_comparison.csv', index=False)
        
        # Save individual model objects (for models that support it)
        for model_name, result in all_results.items():
            if 'ARIMA' in model_name and 'model_info' in result:
                model_file = self.model_dir / f'{model_name.lower()}_model.pkl'
                joblib.dump(result['model_info'], model_file)
        
        print(f"💾 Model results saved to {self.model_dir}")

# Test the modeling pipeline
if __name__ == "__main__":
    forecaster = EconomicForecaster()
    all_results, comparison = forecaster.comprehensive_modeling_pipeline()
    
    print(f"\n🎯 Modeling complete! Check results in the models/ directory.")
