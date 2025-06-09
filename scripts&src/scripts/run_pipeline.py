"""
Complete Economic Forecasting Pipeline
Automated workflow from data collection to dashboard deployment
"""
import sys
import os
from pathlib import Path
import time
import logging
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

# Configure logging
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EconomicForecastingPipeline:
    """Complete automated pipeline for economic forecasting project"""
    
    def __init__(self):
        self.project_root = project_root
        self.start_time = time.time()
        self.pipeline_status = {
            'data_collection': False,
            'preprocessing': False,
            'eda': False,
            'modeling': False,
            'evaluation': False,
            'dashboard_ready': False
        }
        
    def print_banner(self):
        """Print professional pipeline banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║              ECONOMIC FORECASTING PIPELINE v2.0                  ║
║                                                                  ║
║           Macro Research & Model Development Suite               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Starting forecasting workflow...
Pipeline initiated: {timestamp}
Project root: {root}

""".format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            root=self.project_root
        )
        print(banner)
        logger.info("Economic Forecasting Pipeline initiated")
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        logger.info("🔍 Checking dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'statsmodels', 'requests', 'streamlit',
            'joblib', 'scipy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} - OK")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"❌ {package} - MISSING")
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.error("Please install missing packages: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("All dependencies satisfied")
        return True
    
    def setup_directory_structure(self):
        """Ensure all required directories exist"""
        logger.info("📁 Setting up directory structure...")
        
        directories = [
            'data/raw',
            'data/processed',
            'models',
            'results',
            'notebooks/figures',
            'logs',
            'app'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created/verified: {directory}")
        
        logger.info("Directory structure ready")
    
    def run_data_collection(self):
        """Execute data collection from FRED API"""
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("="*60)
        
        try:
            from data_collection import FREDCollector
            
            # Check for API key
            api_key = os.getenv('FRED_API_KEY')
            if not api_key or api_key == 'your_fred_api_key_here':
                logger.warning("⚠️ FRED API key not configured!")
                logger.warning("Please set FRED_API_KEY environment variable")
                logger.warning("Get your free API key at: https://research.stlouisfed.org/")
                
                # Prompt for API key
                api_key = input("\nEnter your FRED API key (or press Enter to skip): ").strip()
                if not api_key:
                    logger.warning("Skipping data collection - using existing data if available")
                    return False
            
            # Economic indicators to collect
            economic_series = {
                'GDP': 'GDPC1',           # Real GDP (Quarterly)
                'UNEMPLOYMENT': 'UNRATE', # Unemployment Rate (Monthly)
                'INFLATION': 'CPIAUCSL',  # Consumer Price Index (Monthly)
                'FED_RATE': 'FEDFUNDS',   # Federal Funds Rate (Monthly)
                'EMPLOYMENT': 'PAYEMS',   # Non-farm Payrolls (Monthly)
                'INDUSTRIAL_PRODUCTION': 'INDPRO',  # Industrial Production Index
                'CONSUMER_SENTIMENT': 'UMCSENT'     # Consumer Sentiment Index
            }
            
            logger.info(f"Collecting {len(economic_series)} economic indicators...")
            
            collector = FREDCollector(api_key)
            economic_data = collector.get_multiple_series(
                economic_series, 
                start_date='2000-01-01'
            )
            
            if not economic_data.empty:
                # Save data with inspection
                collector.inspect_data(economic_data)
                saved_file = collector.save_data_to_csv(economic_data, 'economic_data.csv')
                
                logger.info(f"Data collection completed successfully")
                logger.info(f"Data saved to: {saved_file}")
                self.pipeline_status['data_collection'] = True
                return True
            else:
                logger.error("❌ No data collected")
                return False
                
        except Exception as e:
            logger.error(f"❌ Data collection failed: {str(e)}")
            return False
    
    def run_preprocessing(self):
        """Execute data preprocessing and feature engineering"""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("="*60)
        
        try:
            from preprocessing import EconomicPreprocessor
            
            preprocessor = EconomicPreprocessor()
            processed_data, metadata = preprocessor.preprocess_pipeline(
                filepath='data/raw/economic_data.csv',
                save_results=True
            )
            
            if processed_data is not None and not processed_data.empty:
                logger.info("Preprocessing completed successfully")
                logger.info(f"Final dataset shape: {metadata['final_shape']}")
                logger.info(f"Features created: {metadata['features_created']}")
                logger.info(f"Data loss: {metadata['data_loss_pct']:.1f}%")
                
                # Log transformations applied
                logger.info("\nTransformations Applied:")
                for series, transform in metadata['transformations'].items():
                    logger.info(f"   {series}: {transform['method']}")
                
                self.pipeline_status['preprocessing'] = True
                return True
            else:
                logger.error("❌ Preprocessing failed - no output data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {str(e)}")
            return False
    
    def run_eda(self):
        """Execute exploratory data analysis"""
        logger.info("\n" + "="*60)
        logger.info("📈 STEP 3: EXPLORATORY DATA ANALYSIS")
        logger.info("="*60)
        
        try:
            from visualization import EconomicEDA
            
            eda = EconomicEDA()
            full_data, main_data, correlations = eda.comprehensive_eda_report(
                filepath='data/processed/processed_economic_data.csv'
            )
            
            if full_data is not None and not full_data.empty:
                logger.info("EDA completed successfully")
                logger.info(f"Analysis dataset shape: {full_data.shape}")
                logger.info(f"Main indicators analyzed: {main_data.shape[1]}")
                
                # Log key correlations
                if correlations is not None:
                    logger.info("\nKey Correlations Identified:")
                    # Find strongest correlations
                    corr_values = correlations.values
                    import numpy as np
                    np.fill_diagonal(corr_values, 0)
                    max_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
                    min_idx = np.unravel_index(np.argmin(corr_values), corr_values.shape)
                    
                    max_pair = (correlations.index[max_idx[0]], correlations.columns[max_idx[1]])
                    min_pair = (correlations.index[min_idx[0]], correlations.columns[min_idx[1]])
                    
                    logger.info(f"   Strongest positive: {max_pair[0]} ↔ {max_pair[1]} ({corr_values[max_idx]:.3f})")
                    logger.info(f"   Strongest negative: {min_pair[0]} ↔ {min_pair[1]} ({corr_values[min_idx]:.3f})")
                
                self.pipeline_status['eda'] = True
                return True
            else:
                logger.error("❌ EDA failed - no output data")
                return False
                
        except Exception as e:
            logger.error(f"❌ EDA failed: {str(e)}")
            return False
    
    def run_modeling(self):
        """Execute comprehensive modeling pipeline"""
        logger.info("\n" + "="*60)
        logger.info("STEP 4: MODEL DEVELOPMENT")
        logger.info("="*60)
        
        try:
            from models import EconomicForecaster
            
            forecaster = EconomicForecaster()
            all_results, comparison = forecaster.comprehensive_modeling_pipeline(
                filepath='data/processed/processed_economic_data.csv'
            )
            
            if all_results and comparison is not None:
                logger.info("Modeling completed successfully")
                logger.info(f"Models trained: {len(all_results)}")
                
                if not comparison.empty:
                    best_model = comparison.iloc[0]
                    logger.info(f"Best performing model: {best_model['Model']}")
                    logger.info(f"Best RMSE: {best_model['RMSE']:.4f}")
                    logger.info(f"Best MAE: {best_model['MAE']:.4f}")
                    
                    # Log model rankings
                    logger.info("\nModel Rankings (Top 3):")
                    for i, (_, model) in enumerate(comparison.head(3).iterrows()):
                        logger.info(f"   {i+1}. {model['Model']} (RMSE: {model['RMSE']:.4f})")
                
                self.pipeline_status['modeling'] = True
                return True
            else:
                logger.error("❌ Modeling failed - no results generated")
                return False
                
        except Exception as e:
            logger.error(f"❌ Modeling failed: {str(e)}")
            return False
    
    def run_evaluation(self):
        """Execute comprehensive model evaluation"""
        logger.info("\n" + "="*60)
        logger.info("STEP 5: MODEL EVALUATION")
        logger.info("="*60)
        
        try:
            from evaluation import ModelEvaluator
            
            evaluator = ModelEvaluator()
            results = evaluator.comprehensive_evaluation_pipeline()
            
            if results:
                enhanced_comparison = results['enhanced_comparison']
                best_model = enhanced_comparison.iloc[0]
                
                logger.info("Model evaluation completed successfully")
                logger.info(f"Best model: {best_model['Model']}")
                logger.info(f"Performance category: {best_model['Performance_Category']}")
                logger.info(f"Overall rank: {best_model['Overall_Rank']:.1f}")
                
                # Performance distribution
                performance_dist = enhanced_comparison['Performance_Category'].value_counts()
                logger.info("\nPerformance Distribution:")
                for category, count in performance_dist.items():
                    logger.info(f"   {category}: {count} models")
                
                self.pipeline_status['evaluation'] = True
                return True
            else:
                logger.error("❌ Model evaluation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {str(e)}")
            return False
    
    def prepare_dashboard(self):
        """Prepare dashboard for deployment"""
        logger.info("\n" + "="*60)
        logger.info("STEP 6: DASHBOARD PREPARATION")
        logger.info("="*60)
        
        try:
            # Check if dashboard file exists
            dashboard_path = self.project_root / 'app' / 'dashboard.py'
            
            if dashboard_path.exists():
                logger.info("Dashboard file found")
                
                # Check if all required data files exist
                required_files = [
                    'data/processed/processed_economic_data.csv',
                    'models/model_comparison.csv',
                    'results/enhanced_model_comparison.csv'
                ]
                
                missing_files = []
                for file_path in required_files:
                    full_path = self.project_root / file_path
                    if full_path.exists():
                        logger.info(f"✅ {file_path} - Available")
                    else:
                        missing_files.append(file_path)
                        logger.warning(f"⚠️ {file_path} - Missing")
                
                if not missing_files:
                    logger.info("Dashboard ready for deployment")
                    logger.info(f"Run dashboard with: streamlit run {dashboard_path}")
                    self.pipeline_status['dashboard_ready'] = True
                    return True
                else:
                    logger.warning(f"⚠️ Dashboard missing {len(missing_files)} required files")
                    return False
            else:
                logger.error("❌ Dashboard file not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Dashboard preparation failed: {str(e)}")
            return False
    
    def generate_pipeline_report(self):
        """Generate comprehensive pipeline execution report"""
        logger.info("\n" + "="*60)
        logger.info("PIPELINE EXECUTION REPORT")
        logger.info("="*60)
        
        # Calculate execution time
        execution_time = time.time() - self.start_time
        
        # Generate report
        report = f"""
ECONOMIC FORECASTING PIPELINE EXECUTION REPORT
==============================================

Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Execution Time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)
Project Directory: {self.project_root}

PIPELINE STATUS:
---------------
"""
        
        # Add status for each step
        step_names = {
            'data_collection': 'Data Collection',
            'preprocessing': 'Data Preprocessing', 
            'eda': 'Exploratory Data Analysis',
            'modeling': 'Model Development',
            'evaluation': 'Model Evaluation',
            'dashboard_ready': 'Dashboard Preparation'
        }
        
        for step, status in self.pipeline_status.items():
            status_icon = "✅" if status else "❌"
            report += f"{status_icon} {step_names[step]}: {'COMPLETED' if status else 'FAILED'}\n"
        
        # Overall success rate
        completed_steps = sum(self.pipeline_status.values())
        total_steps = len(self.pipeline_status)
        success_rate = (completed_steps / total_steps) * 100
        
        report += f"""
OVERALL PIPELINE SUCCESS: {completed_steps}/{total_steps} steps ({success_rate:.1f}%)

NEXT STEPS:
----------
"""
        
        if self.pipeline_status['dashboard_ready']:
            report += """✅ Pipeline completed successfully!
Run the dashboard: streamlit run app/dashboard.py
View results in: results/ directory
Check figures in: notebooks/figures/ directory
Review logs in: logs/ directory
"""
        else:
            failed_steps = [step for step, status in self.pipeline_status.items() if not status]
            report += f"""⚠️ Pipeline partially completed
Failed steps: {', '.join(failed_steps)}
Check logs for error details
Re-run pipeline to retry failed steps
"""
        
        # Save report
        report_path = self.project_root / 'results' / f'pipeline_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        logger.info(f"📋 Pipeline report saved to: {report_path}")
        
        return success_rate >= 80  # Consider pipeline successful if 80%+ steps completed
    
    def run_complete_pipeline(self, skip_on_failure=False):
        """Execute the complete economic forecasting pipeline"""
        self.print_banner()
        
        # Check dependencies first
        if not self.check_dependencies():
            logger.error("❌ Pipeline aborted due to missing dependencies")
            return False
        
        # Setup directory structure
        self.setup_directory_structure()
        
        # Execute pipeline steps
        pipeline_steps = [
            ('data_collection', self.run_data_collection),
            ('preprocessing', self.run_preprocessing),
            ('eda', self.run_eda),
            ('modeling', self.run_modeling),
            ('evaluation', self.run_evaluation),
            ('dashboard_ready', self.prepare_dashboard)
        ]
        
        for step_name, step_function in pipeline_steps:
            try:
                success = step_function()
                
                if not success and not skip_on_failure:
                    logger.error(f"❌ Pipeline stopped at step: {step_name}")
                    break
                elif not success:
                    logger.warning(f"⚠️ Step {step_name} failed but continuing...")
                    
            except KeyboardInterrupt:
                logger.warning("⚠️ Pipeline interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in {step_name}: {str(e)}")
                if not skip_on_failure:
                    break
        
        # Generate final report
        success = self.generate_pipeline_report()
        
        if success:
            logger.info("Pipeline completed successfully!")
        else:
            logger.warning("⚠️ Pipeline completed with issues")
        
        return success

def main():
    """Main pipeline execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Economic Forecasting Pipeline')
    parser.add_argument('--skip-on-failure', action='store_true',
                       help='Continue pipeline even if individual steps fail')
    parser.add_argument('--step', choices=['data', 'preprocess', 'eda', 'model', 'eval', 'dashboard'],
                       help='Run only a specific pipeline step')
    
    args = parser.parse_args()
    
    pipeline = EconomicForecastingPipeline()
    
    if args.step:
        # Run specific step
        step_mapping = {
            'data': pipeline.run_data_collection,
            'preprocess': pipeline.run_preprocessing,
            'eda': pipeline.run_eda,
            'model': pipeline.run_modeling,
            'eval': pipeline.run_evaluation,
            'dashboard': pipeline.prepare_dashboard
        }
        
        pipeline.print_banner()
        pipeline.check_dependencies()
        pipeline.setup_directory_structure()
        
        success = step_mapping[args.step]()
        if success:
            logger.info(f"✅ Step '{args.step}' completed successfully")
        else:
            logger.error(f"❌ Step '{args.step}' failed")
    else:
        # Run complete pipeline
        pipeline.run_complete_pipeline(skip_on_failure=args.skip_on_failure)

if __name__ == "__main__":
    main()
