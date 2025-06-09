"""
Economic Forecasting Dashboard
Interactive web application for model deployment and visualization
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Economic Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EconomicForecastingDashboard:
    """Dashboard for economic forecasting models"""
    
    def __init__(self):
        self.data_dir = Path('data')
        self.models_dir = Path('models')
        self.results_dir = Path('results')
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
    
    def load_data(self):
        """Load all necessary data for the dashboard"""
        try:
            # Load processed economic data
            processed_data_path = self.data_dir / 'processed' / 'processed_economic_data.csv'
            if processed_data_path.exists():
                self.processed_data = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
                
                # Extract main indicators
                self.main_indicators = [col for col in self.processed_data.columns if not any(
                    suffix in col for suffix in ['_lag_', '_ma_', '_std_']
                )]
                
                st.session_state.data_loaded = True
                return True
            else:
                st.error("Processed data not found. Please run the preprocessing pipeline first.")
                return False
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def load_model_results(self):
        """Load model comparison and evaluation results"""
        try:
            # Load model comparison
            comparison_path = self.models_dir / 'model_comparison.csv'
            if comparison_path.exists():
                self.model_comparison = pd.read_csv(comparison_path)
                
            # Load enhanced comparison if available
            enhanced_path = self.results_dir / 'enhanced_model_comparison.csv'
            if enhanced_path.exists():
                self.enhanced_comparison = pd.read_csv(enhanced_path)
            else:
                self.enhanced_comparison = self.model_comparison
                
            st.session_state.models_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading model results: {str(e)}")
            return False
    
    def create_header(self):
        """Create professional dashboard header"""
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1f4e79 0%, #2e7bb8 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center; margin: 0;">
                 Economic Forecasting Dashboard
            </h1>
            <p style="color: #e6f3ff; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                 Macro Research & Model Deployment Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def create_sidebar(self):
        """Create interactive sidebar with controls"""
        st.sidebar.markdown("## 📊 Dashboard Controls")
        
        # Data loading status
        st.sidebar.markdown("### Data Status")
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data Loaded")
        else:
            st.sidebar.error("❌ Data Not Loaded")
            
        if st.session_state.models_loaded:
            st.sidebar.success("✅ Models Loaded")
        else:
            st.sidebar.error("❌ Models Not Loaded")
        
        # Refresh data button
        if st.sidebar.button("🔄 Refresh Data"):
            self.load_data()
            self.load_model_results()
        
        st.sidebar.markdown("---")
        
        # Navigation
        st.sidebar.markdown("### Navigation")
        page = st.sidebar.selectbox(
            "Select Dashboard Page",
            ["Overview", "Model Performance", "Forecasting", "Technical Details"]
        )
        
        return page
    
    def overview_page(self):
        """Create overview dashboard page"""
        st.markdown("## Economic Data Overview")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first using the sidebar controls.")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Total Indicators",
                value=len(self.main_indicators),
                delta="Main Economic Variables"
            )
        
        with col2:
            st.metric(
                label="📅 Data Points",
                value=len(self.processed_data),
                delta=f"From {self.processed_data.index.min().strftime('%Y-%m')}"
            )
        
        with col3:
            st.metric(
                label="🔧 Features Created",
                value=self.processed_data.shape[1] - len(self.main_indicators),
                delta="Engineered Features"
            )
        
        with col4:
            if st.session_state.models_loaded:
                best_model = self.enhanced_comparison.iloc[0]['Model']
                best_rmse = self.enhanced_comparison.iloc[0]['RMSE']
                st.metric(
                    label="🏆 Best Model",
                    value=best_model.split('_')[0],
                    delta=f"RMSE: {best_rmse:.4f}"
                )
        
        # Interactive time series plot
        st.markdown("### Economic Indicators Time Series")
        
        # Indicator selection
        selected_indicators = st.multiselect(
            "Select indicators to display:",
            self.main_indicators,
            default=self.main_indicators[:3]
        )
        
        if selected_indicators:
            # Create interactive plotly chart
            fig = make_subplots(
                rows=len(selected_indicators), cols=1,
                subplot_titles=selected_indicators,
                vertical_spacing=0.05
            )
            
            for i, indicator in enumerate(selected_indicators):
                fig.add_trace(
                    go.Scatter(
                        x=self.processed_data.index,
                        y=self.processed_data[indicator],
                        name=indicator,
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(
                height=200*len(selected_indicators),
                title="Economic Indicators (Transformed Data)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔗 Indicator Correlations")
        
        correlation_matrix = self.processed_data[self.main_indicators].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Economic Indicators Correlation Matrix"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    def model_performance_page(self):
        """Create model performance comparison page"""
        st.markdown("## Model Performance Analysis")
        
        if not st.session_state.models_loaded:
            st.warning("Please load model results first.")
            return
        
        # Performance summary
        st.markdown("### 🏆 Model Rankings")
        
        # Display enhanced comparison table
        st.dataframe(
            self.enhanced_comparison[['Model', 'RMSE', 'MAE', 'Performance_Category', 'Overall_Rank']],
            use_container_width=True
        )
        
        # Performance visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE comparison
            fig_rmse = px.bar(
                self.enhanced_comparison,
                x='Model',
                y='RMSE',
                title="Root Mean Squared Error by Model",
                color='Performance_Category',
                color_discrete_map={
                    'Excellent': '#2E8B57',
                    'Good': '#4682B4', 
                    'Fair': '#DAA520',
                    'Poor': '#CD5C5C'
                }
            )
            fig_rmse.update_xaxis(tickangle=45)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            # Performance category distribution
            category_counts = self.enhanced_comparison['Performance_Category'].value_counts()
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Model Performance Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed metrics comparison
        st.markdown("### 📋 Detailed Performance Metrics")
        
        # Radar chart for top 3 models
        top_models = self.enhanced_comparison.head(3)
        
        if len(top_models) > 0:
            metrics = ['RMSE', 'MAE']
            available_metrics = [m for m in metrics if m in top_models.columns]
            
            if available_metrics:
                fig_radar = go.Figure()
                
                for _, model in top_models.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[1/model[metric] for metric in available_metrics],  # Inverse for radar chart
                        theta=available_metrics,
                        fill='toself',
                        name=model['Model']
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max([1/top_models[m].min() for m in available_metrics])]
                        )),
                    title="Top 3 Models Performance Comparison (Higher = Better)"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
    
    def forecasting_page(self):
        """Create interactive forecasting page"""
        st.markdown("## 🔮 Economic Forecasting Interface")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first.")
            return
        
        # Forecasting controls
        st.markdown("### ⚙️ Forecast Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_horizon = st.slider(
                "Forecast Horizon (periods)",
                min_value=1,
                max_value=24,
                value=12,
                help="Number of periods to forecast ahead"
            )
        
        with col2:
            confidence_level = st.selectbox(
                "Confidence Level",
                [0.90, 0.95, 0.99],
                index=1,
                help="Confidence interval for forecasts"
            )
        
        with col3:
            selected_indicator = st.selectbox(
                "Economic Indicator",
                self.main_indicators,
                help="Select indicator to forecast"
            )
        
        # Generate forecast button
        if st.button("🚀 Generate Forecast", type="primary"):
            self.generate_forecast_demo(selected_indicator, forecast_horizon, confidence_level)
    
    def generate_forecast_demo(self, indicator, horizon, confidence):
        """Generate demonstration forecast"""
        st.markdown("### Forecast Results")
        
        # Get recent data for the selected indicator
        recent_data = self.processed_data[indicator].dropna().tail(100)
        
        # Simple demonstration forecast (in practice, would use trained models)
        last_values = recent_data.tail(12).values
        trend = np.mean(np.diff(last_values))
        
        # Generate forecast
        forecast_dates = pd.date_range(
            start=recent_data.index[-1] + timedelta(days=30),
            periods=horizon,
            freq='M'
        )
        
        # Simple trend + noise forecast for demonstration
        forecast_values = []
        last_value = recent_data.iloc[-1]
        
        for i in range(horizon):
            next_value = last_value + trend + np.random.normal(0, recent_data.std() * 0.1)
            forecast_values.append(next_value)
            last_value = next_value
        
        # Calculate confidence intervals
        forecast_std = recent_data.std()
        z_score = 1.96 if confidence == 0.95 else (1.645 if confidence == 0.90 else 2.576)
        
        upper_bound = np.array(forecast_values) + z_score * forecast_std
        lower_bound = np.array(forecast_values) - z_score * forecast_std
        
        # Create forecast visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data.values,
            mode='lines',
            name='Historical Data',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0),
            name=f'{int(confidence*100)}% Confidence Interval'
        ))
        
        fig.update_layout(
            title=f'{indicator} Forecast - {horizon} Periods Ahead',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary table
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_values,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound
        })
        
        st.markdown("### Forecast Summary")
        st.dataframe(forecast_df, use_container_width=True)
    
    def technical_details_page(self):
        """Create technical documentation page"""
        st.markdown("## Technical Implementation Details")
        
        # Project overview
        st.markdown("""
        ### Project Overview
        
        This economic forecasting system demonstrates a complete machine learning pipeline for macro research:
        
        **Key Components:**
        - **Data Collection**: FRED API integration for economic indicators
        - **Preprocessing**: Stationarity testing, feature engineering, missing value handling
        - **Modeling**: ARIMA, VAR, LSTM, and hybrid approaches
        - **Evaluation**: Comprehensive metrics and cross-validation
        - **Deployment**: Interactive dashboard with real-time forecasting
        """)
        
        # Technical architecture
        st.markdown("""
        ### Technical Architecture
        
        **Data Pipeline:**
        ```
        FRED API → Raw Data → Preprocessing → Feature Engineering → Model Training
                                     ↓
        Dashboard ← Model Deployment ← Model Evaluation ← Trained Models
        ```
        
        **Model Types Implemented:**
        - **ARIMA**: Traditional econometric approach for univariate forecasting
        - **VAR**: Multivariate analysis capturing economic interdependencies  
        - **LSTM**: Deep learning for non-linear pattern recognition
        - **Hybrid**: ARIMA-LSTM combination for optimal performance
        """)
        
        # Performance metrics explanation
        st.markdown("""
        ### Evaluation Methodology
        
        **Key Metrics:**
        - **RMSE**: Root Mean Squared Error - penalizes large forecast errors
        - **MAE**: Mean Absolute Error - robust to outliers
        - **MAPE**: Mean Absolute Percentage Error - scale-independent
        - **Directional Accuracy**: Critical for trading/policy decisions
        - **Theil's U**: Performance relative to naive forecast
        
        **Validation Approach:**
        - Time series cross-validation (walk-forward analysis)
        - Out-of-sample testing maintaining temporal order
        - Statistical significance testing (Diebold-Mariano)
        """)
        
        # Banking relevance
        st.markdown("""
        ### Banking & Finance Applications
        
        **Regulatory Compliance:**
        - CECL (Current Expected Credit Loss) modeling
        - Stress testing scenarios
        - Risk appetite framework
        
        **Business Applications:**
        - Economic scenario generation
        - Credit risk assessment
        - Strategic planning and budgeting
        - Investment decision support
        """)
        
        # Code repository info
        if st.button("📂 View Project Structure"):
            st.code("""
            economic_forecasting/
            ├── src/
            │   ├── data_collection.py    # FRED API integration
            │   ├── preprocessing.py      # Data cleaning & feature engineering
            │   ├── models.py            # ARIMA, VAR, LSTM implementations
            │   ├── evaluation.py        # Model assessment & validation
            │   └── visualization.py     # EDA and analysis plots
            ├── app/
            │   └── dashboard.py         # Streamlit deployment
            ├── data/
            │   ├── raw/                 # Original FRED data
            │   └── processed/           # Cleaned & engineered features
            ├── models/                  # Trained model artifacts
            ├── results/                 # Evaluation reports
            └── notebooks/               # Jupyter analysis notebooks
            """, language="text")
    
    def run_dashboard(self):
        """Main dashboard execution"""
        # Load data on startup
        if not st.session_state.data_loaded:
            self.load_data()
        if not st.session_state.models_loaded:
            self.load_model_results()
        
        # Create header
        self.create_header()
        
        # Create sidebar and get selected page
        selected_page = self.create_sidebar()
        
        # Route to appropriate page
        if selected_page == "Overview":
            self.overview_page()
        elif selected_page == "Model Performance":
            self.model_performance_page()
        elif selected_page == "Forecasting":
            self.forecasting_page()
        elif selected_page == "Technical Details":
            self.technical_details_page()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Economic Forecasting Dashboard | Professional Macro Research Portfolio</p>
            <p>Built with Python, Streamlit, and Advanced Time Series Modeling</p>
        </div>
        """, unsafe_allow_html=True)

# Run the dashboard
if __name__ == "__main__":
    dashboard = EconomicForecastingDashboard()
    dashboard.run_dashboard()
