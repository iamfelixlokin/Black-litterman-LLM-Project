"""
Main execution script for Black-Litterman portfolio optimization
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import setup_logging
from data_collection import DataCollector
from llm_view_generator import LLMViewGenerator
from backtest_engine import BacktestEngine
from performance_metrics import PerformanceAnalyzer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collect_data(config: dict, force_update: bool = False):
    """
    Collect price and news data
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    force_update : bool
        Force data collection even if cached data exists
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    prices_file = os.path.join(data_dir, 'prices.csv')
    
    # Check if data already exists
    if os.path.exists(prices_file) and not force_update:
        logger.info("Loading cached price data...")
        collector = DataCollector(config['assets']['tickers'], config['assets']['benchmark'])
        prices = collector.load_data(prices_file)
        return prices, collector
    
    logger.info("Collecting fresh data...")
    
    # Initialize collector
    collector = DataCollector(
        config['assets']['tickers'],
        config['assets']['benchmark']
    )
    
    # Fetch price data
    prices = collector.fetch_price_data(
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date']
    )
    
    # Save data
    collector.save_data(prices, prices_file)
    
    return prices, collector


def setup_llm_generator(config: dict) -> LLMViewGenerator:
    """
    Setup LLM view generator
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    LLMViewGenerator or None
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not found. LLM views will be disabled.")
        logger.warning("Set ANTHROPIC_API_KEY environment variable to enable LLM-generated views.")
        return None
    
    logger.info("Initializing LLM view generator...")
    
    generator = LLMViewGenerator(
        api_key=api_key,
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens'],
        confidence_omega=config['black_litterman'].get('confidence_omega', None)
    )
    
    return generator


def run_backtest(config: dict, 
                prices: pd.DataFrame,
                llm_generator: LLMViewGenerator = None,
                data_collector: DataCollector = None):
    """
    Run complete backtest with all strategies
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    prices : pd.DataFrame
        Price data
    llm_generator : LLMViewGenerator, optional
        LLM view generator
    data_collector : DataCollector, optional
        Data collector for context generation
    
    Returns:
    --------
    dict : Backtest results
    """
    logger.info("Initializing backtest engine...")
    
    # Separate asset prices and benchmark
    all_tickers = config['assets']['tickers']
    benchmark = config['assets']['benchmark']
    
    # 移除 SPY 從資產列表（但保留在 prices 中供 Alpha 計算用）
    asset_tickers = [t for t in all_tickers if t != benchmark]
    
    asset_prices = prices[asset_tickers]
    benchmark_prices = prices[benchmark]
    
    # Initialize backtest engine
    engine = BacktestEngine(asset_prices, benchmark_prices, config)
    engine.full_prices = prices  # 傳入完整 prices（包含 SPY）
    
    # Run all strategies
    logger.info("\n" + "="*80)
    logger.info("RUNNING BACKTEST FOR ALL STRATEGIES")
    logger.info("="*80)
    
    results = engine.run_all_strategies(llm_generator, data_collector)
    
    return results


def analyze_results(results: dict, 
                   config: dict,
                   output_dir: str = None):
    """
    Analyze and visualize results
    
    Parameters:
    -----------
    results : dict
        Backtest results
    config : dict
        Configuration dictionary
    output_dir : str, optional
        Output directory for results
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Analyzing results...")
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(risk_free_rate=config['metrics']['risk_free_rate'])
    
    # Get benchmark returns
    benchmark_returns = results['spy_benchmark']['portfolio_values']['returns']
    
    # Generate comprehensive report
    analyzer.generate_report(results, benchmark_returns, output_dir)
    
    # Save detailed results
    import pickle
    results_file = os.path.join(output_dir, 'backtest_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Detailed results saved to {results_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Black-Litterman Portfolio Optimization with LLM Views'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--force-update',
        action='store_true',
        help='Force data collection even if cached data exists'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM view generation (faster testing)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    setup_logging(log_file=log_file, level="INFO")
    
    logger.info("="*80)
    logger.info("BLACK-LITTERMAN PORTFOLIO OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config file: {args.config}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
        
        # Collect data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("="*80)
        prices, data_collector = collect_data(config, args.force_update)
        logger.info(f"Price data shape: {prices.shape}")
        logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        
        # Setup LLM generator
        logger.info("\n" + "="*80)
        logger.info("STEP 2: LLM SETUP")
        logger.info("="*80)
        
        if args.no_llm:
            logger.info("LLM view generation disabled (--no-llm flag)")
            llm_generator = None
        else:
            llm_generator = setup_llm_generator(config)
        
        # Run backtest
        logger.info("\n" + "="*80)
        logger.info("STEP 3: BACKTEST EXECUTION")
        logger.info("="*80)
        
        results = run_backtest(config, prices, llm_generator, data_collector)
        
        # Analyze results
        logger.info("\n" + "="*80)
        logger.info("STEP 4: PERFORMANCE ANALYSIS")
        logger.info("="*80)
        
        analyze_results(results, config, args.output_dir)
        
        logger.info("\n" + "="*80)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results saved to: {args.output_dir or '../results'}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
