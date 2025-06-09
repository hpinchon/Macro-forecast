"""
Exploratory Data Analysis for Economic Time Series
Comprehensive visualization and pattern analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EconomicEDA:
    """Comprehensive EDA for economic time series data"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.figures_dir = Path('notebooks/figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_processed_data(self, filepath='data/processed/processed_economic_data.csv'):
        """Load the preprocessed economic data"""
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"✅ Loaded processed data: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        return df
    
    def basic_statistics(self, df):
        """
        Generate comprehensive basic statistics
        
        Args:
            df (pd.DataFrame): Economic data
        """
        print("📊 BASIC STATISTICS ANALYSIS")
        print("=" * 50)
        
        # Focus on main economic indicators (not lagged/rolling features)
        main_indicators = [col for col in df.columns if not any(
            suffix in col for suffix in ['_lag_', '_ma_', '_std_']
        )]
        
        main_df = df[main_indicators]
        
        print(f"\n📈 Main Economic Indicators: {main_indicators}")
        print(f"📊 Data shape: {main_df.shape}")
        
        # Descriptive statistics
        print(f"\n📋 Descriptive Statistics:")
        print(main_df.describe().round(4))
        
        # Missing values check
        missing = main_df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n❓ Missing values:")
            for col, count in missing[missing > 0].items():
                print(f"   {col}: {count}")
        else:
            print(f"\n✅ No missing values in main indicators")
        
        # Correlation analysis
        print(f"\n🔗 Correlation Matrix:")
        correlation_matrix = main_df.corr()
        print(correlation_matrix.round(3))
        
        return main_df, correlation_matrix
    
    def plot_time_series(self, df, save_fig=True):
        """
        Plot individual time series for main economic indicators
        
        Args:
            df (pd.DataFrame): Economic data
            save_fig (bool): Whether to save the figure
        """
        # Focus on main indicators
        main_indicators = [col for col in df.columns if not any(
            suffix in col for suffix in ['_lag_', '_ma_', '_std_']
        )]
        
        n_indicators = len(main_indicators)
        fig, axes = plt.subplots(n_indicators, 1, figsize=(15, 3*n_indicators))
        
        if n_indicators == 1:
            axes = [axes]
        
        for i, indicator in enumerate(main_indicators):
            axes[i].plot(df.index, df[indicator], linewidth=2, alpha=0.8)
            axes[i].set_title(f'{indicator} (Transformed)', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Value', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add trend line
            x_numeric = np.arange(len(df[indicator].dropna()))
            y_values = df[indicator].dropna().values
            if len(y_values) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
                trend_line = slope * x_numeric + intercept
                axes[i].plot(df[indicator].dropna().index, trend_line, 
                           '--', color='red', alpha=0.7, label=f'Trend (R²={r_value**2:.3f})')
                axes[i].legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(self.figures_dir / 'time_series_plots.png', 
                       dpi=300, bbox_inches='tight')
            print(f"📊 Time series plots saved to {self.figures_dir / 'time_series_plots.png'}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, correlation_matrix, save_fig=True):
        """
        Create correlation heatmap for economic indicators
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            save_fig (bool): Whether to save the figure
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Economic Indicators Correlation Matrix', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(self.figures_dir / 'correlation_heatmap.png', 
                       dpi=300, bbox_inches='tight')
            print(f"📊 Correlation heatmap saved to {self.figures_dir / 'correlation_heatmap.png'}")
        
        plt.show()
        
        # Interpret correlations
        print(f"\n🔍 CORRELATION INSIGHTS:")
        print("=" * 30)
        
        # Find strongest positive and negative correlations
        corr_values = correlation_matrix.values
        np.fill_diagonal(corr_values, 0)  # Remove self-correlations
        
        max_corr_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
        min_corr_idx = np.unravel_index(np.argmin(corr_values), corr_values.shape)
        
        max_corr = corr_values[max_corr_idx]
        min_corr = corr_values[min_corr_idx]
        
        max_pair = (correlation_matrix.index[max_corr_idx[0]], 
                   correlation_matrix.columns[max_corr_idx[1]])
        min_pair = (correlation_matrix.index[min_corr_idx[0]], 
                   correlation_matrix.columns[min_corr_idx[1]])
        
        print(f"🔺 Strongest positive correlation: {max_pair[0]} ↔ {max_pair[1]} ({max_corr:.3f})")
        print(f"🔻 Strongest negative correlation: {min_pair[0]} ↔ {min_pair[1]} ({min_corr:.3f})")
    
    def seasonal_decomposition_analysis(self, df, save_fig=True):
        """
        Perform seasonal decomposition on economic indicators
        
        Args:
            df (pd.DataFrame): Economic data
            save_fig (bool): Whether to save figures
        """
        print(f"\n📈 SEASONAL DECOMPOSITION ANALYSIS")
        print("=" * 40)
        
        main_indicators = [col for col in df.columns if not any(
            suffix in col for suffix in ['_lag_', '_ma_', '_std_']
        )]
        
        for indicator in main_indicators:
            series = df[indicator].dropna()
            
            if len(series) < 24:  # Need sufficient data for decomposition
                print(f"⚠️ {indicator}: Insufficient data for decomposition")
                continue
            
            try:
                # Determine period based on data frequency
                freq = pd.infer_freq(series.index)
                if freq and 'M' in freq:
                    period = 12  # Monthly data
                elif freq and 'Q' in freq:
                    period = 4   # Quarterly data
                else:
                    period = 12  # Default to annual cycle
                
                # Perform decomposition
                decomposition = seasonal_decompose(series, 
                                                 model='additive', 
                                                 period=period)
                
                # Plot decomposition
                fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                
                decomposition.observed.plot(ax=axes[0], title=f'{indicator} - Original')
                decomposition.trend.plot(ax=axes[1], title='Trend')
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
                decomposition.resid.plot(ax=axes[3], title='Residual')
                
                for ax in axes:
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='x', rotation=45)
                
                plt.suptitle(f'Seasonal Decomposition: {indicator}', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                if save_fig:
                    plt.savefig(self.figures_dir / f'decomposition_{indicator}.png', 
                               dpi=300, bbox_inches='tight')
                
                plt.show()
                
                # Calculate component statistics
                trend_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.trend.dropna() + decomposition.resid.dropna())
                seasonal_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.seasonal.dropna() + decomposition.resid.dropna())
                
                print(f"📊 {indicator}:")
                print(f"   Trend strength: {trend_strength:.3f}")
                print(f"   Seasonal strength: {seasonal_strength:.3f}")
                
            except Exception as e:
                print(f"❌ Error decomposing {indicator}: {str(e)}")
    
    def autocorrelation_analysis(self, df, max_lags=20, save_fig=True):
        """
        Analyze autocorrelation and partial autocorrelation
        
        Args:
            df (pd.DataFrame): Economic data
            max_lags (int): Maximum number of lags to analyze
            save_fig (bool): Whether to save figures
        """
        print(f"\n📊 AUTOCORRELATION ANALYSIS")
        print("=" * 35)
        
        main_indicators = [col for col in df.columns if not any(
            suffix in col for suffix in ['_lag_', '_ma_', '_std_']
        )]
        
        for indicator in main_indicators:
            series = df[indicator].dropna()
            
            if len(series) < max_lags + 10:
                print(f"⚠️ {indicator}: Insufficient data for autocorrelation analysis")
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # ACF plot
            plot_acf(series, lags=max_lags, ax=axes[0], alpha=0.05)
            axes[0].set_title(f'Autocorrelation Function: {indicator}')
            axes[0].grid(True, alpha=0.3)
            
            # PACF plot
            plot_pacf(series, lags=max_lags, ax=axes[1], alpha=0.05)
            axes[1].set_title(f'Partial Autocorrelation Function: {indicator}')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_fig:
                plt.savefig(self.figures_dir / f'autocorr_{indicator}.png', 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            
            # Calculate Ljung-Box test for white noise
            from statsmodels.stats.diagnostic import acorr_ljungbox
            ljung_box = acorr_ljungbox(series, lags=10, return_df=True)
            
            print(f"📊 {indicator} Autocorrelation Summary:")
            print(f"   Ljung-Box test p-value: {ljung_box['lb_pvalue'].iloc[-1]:.4f}")
            if ljung_box['lb_pvalue'].iloc[-1] < 0.05:
                print(f"   ✅ Significant autocorrelation detected")
            else:
                print(f"   ❌ No significant autocorrelation (white noise)")
    
    def distribution_analysis(self, df, save_fig=True):
        """
        Analyze distributions of economic indicators
        
        Args:
            df (pd.DataFrame): Economic data
            save_fig (bool): Whether to save figures
        """
        print(f"\n📊 DISTRIBUTION ANALYSIS")
        print("=" * 30)
        
        main_indicators = [col for col in df.columns if not any(
            suffix in col for suffix in ['_lag_', '_ma_', '_std_']
        )]
        
        n_indicators = len(main_indicators)
        fig, axes = plt.subplots(2, n_indicators, figsize=(4*n_indicators, 8))
        
        if n_indicators == 1:
            axes = axes.reshape(2, 1)
        
        for i, indicator in enumerate(main_indicators):
            series = df[indicator].dropna()
            
            # Histogram
            axes[0, i].hist(series, bins=30, alpha=0.7, density=True)
            axes[0, i].set_title(f'{indicator} Distribution')
            axes[0, i].set_ylabel('Density')
            axes[0, i].grid(True, alpha=0.3)
            
            # Add normal distribution overlay
            mu, sigma = series.mean(), series.std()
            x = np.linspace(series.min(), series.max(), 100)
            axes[0, i].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', 
                           label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
            axes[0, i].legend()
            
            # Q-Q plot
            stats.probplot(series, dist="norm", plot=axes[1, i])
            axes[1, i].set_title(f'{indicator} Q-Q Plot')
            axes[1, i].grid(True, alpha=0.3)
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(series)
            jb_stat, jb_p = stats.jarque_bera(series)
            
            print(f"📊 {indicator} Distribution Tests:")
            print(f"   Shapiro-Wilk p-value: {shapiro_p:.4f}")
            print(f"   Jarque-Bera p-value: {jb_p:.4f}")
            print(f"   Skewness: {stats.skew(series):.3f}")
            print(f"   Kurtosis: {stats.kurtosis(series):.3f}")
            
            if shapiro_p > 0.05 and jb_p > 0.05:
                print(f"   ✅ Approximately normal distribution")
            else:
                print(f"   ❌ Non-normal distribution")
            print()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(self.figures_dir / 'distribution_analysis.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def comprehensive_eda_report(self, filepath='data/processed/processed_economic_data.csv'):
        """
        Generate comprehensive EDA report
        
        Args:
            filepath (str): Path to processed data
        """
        print("🚀 COMPREHENSIVE ECONOMIC DATA ANALYSIS")
        print("=" * 60)
        
        # Load data
        df = self.load_processed_data(filepath)
        
        # Basic statistics
        main_df, correlation_matrix = self.basic_statistics(df)
        
        # Time series plots
        print(f"\n📈 GENERATING TIME SERIES PLOTS...")
        self.plot_time_series(main_df)
        
        # Correlation analysis
        print(f"\n🔗 CORRELATION ANALYSIS...")
        self.plot_correlation_heatmap(correlation_matrix)
        
        # Seasonal decomposition
        self.seasonal_decomposition_analysis(main_df)
        
        # Autocorrelation analysis
        self.autocorrelation_analysis(main_df)
        
        # Distribution analysis
        self.distribution_analysis(main_df)
        
        print(f"\n✅ EDA COMPLETE!")
        print(f"📁 All figures saved to: {self.figures_dir}")
        print(f"🎯 Ready for time series modeling!")
        
        return df, main_df, correlation_matrix

# Test the EDA module
if __name__ == "__main__":
    eda = EconomicEDA()
    full_data, main_data, correlations = eda.comprehensive_eda_report()
    
    print(f"\n📋 EDA Summary:")
    print(f"   Full dataset shape: {full_data.shape}")
    print(f"   Main indicators shape: {main_data.shape}")
    print(f"   Figures generated: {len(list(eda.figures_dir.glob('*.png')))}")
