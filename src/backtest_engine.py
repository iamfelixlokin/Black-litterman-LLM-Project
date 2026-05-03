"""
Backtest Engine for Portfolio Strategies
Ensures no look-ahead bias with proper time-series split
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import logging
from tqdm import tqdm
import yfinance as yf

from utils import (
    calculate_returns, get_rebalance_dates, calculate_turnover,
    apply_position_limits, ensure_no_lookahead
)
from black_litterman import BlackLittermanModel
from baseline_strategies import BaselineStrategies
from llm_view_generator import LLMViewGenerator
from data_collection import DataCollector

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting framework with strict no-lookahead bias
    """
    
    def __init__(self,
                 prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 config: Dict):
        """
        Initialize backtest engine
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data for assets (datetime index, tickers as columns)
        benchmark_prices : pd.Series
            Benchmark price data
        config : Dict
            Configuration dictionary
        """
        self.prices = prices
        self.benchmark_prices = benchmark_prices
        self.config = config
        # Store full prices for Alpha calculation (including SPY)
        self.full_prices = None  # Will be set by main.py
        
        # Calculate returns
        self.returns = calculate_returns(prices)
        self.benchmark_returns = calculate_returns(benchmark_prices.to_frame()).iloc[:, 0]
        
        # Extract config parameters
        self.initial_capital = config['backtest']['initial_capital']
        self.transaction_cost = config['backtest']['transaction_cost']
        self.slippage = config['backtest']['slippage']
        self.rebalance_frequency = config['backtest']['rebalance_frequency']
        self.lookback_period = config['backtest']['lookback_period']
        
        # Risk management
        self.min_weight = config['risk_management']['min_position_size']
        self.max_weight = config['risk_management']['max_position_size']
        
        # Initialize models
        self.bl_model = BlackLittermanModel(
            risk_aversion=config['black_litterman']['risk_aversion'],
            tau=config['black_litterman']['tau']
        )
        self.baseline = BaselineStrategies(
            risk_free_rate=config['metrics']['risk_free_rate']
        )
        
        # Results storage
        self.results = {}

        # Pre-fetch current market caps once（用於推算歷史市值）
        self._current_market_caps = self._fetch_current_market_caps(prices.columns.tolist())
        
    def _fetch_current_market_caps(self, tickers: List[str]) -> Dict[str, float]:
        """從 yfinance 抓取當前市值（十億美元），回測開始時執行一次。"""
        market_caps = {}
        fallback = 1000.0
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                mc = info.get("marketCap")
                market_caps[ticker] = mc / 1e9 if mc and mc > 0 else fallback
            except Exception:
                market_caps[ticker] = fallback
        logger.info(f"Fetched current market caps: { {k: f'${v:.0f}B' for k, v in market_caps.items()} }")
        return market_caps

    def _get_historical_market_caps(self, tickers: List[str], date: pd.Timestamp) -> np.ndarray:
        """
        用股價比例推算歷史市值（無需額外 API 呼叫）。
        historical_mc ≈ current_mc × (price_at_date / current_price)
        """
        current_prices = self.prices.iloc[-1]   # 最新一筆
        try:
            hist_prices = self.prices.loc[:date].iloc[-1]
        except Exception:
            hist_prices = current_prices

        market_caps = []
        for ticker in tickers:
            mc = self._current_market_caps.get(ticker, 1000.0)
            cur_p = current_prices.get(ticker, 1.0)
            hist_p = hist_prices.get(ticker, cur_p)
            if cur_p > 0:
                mc = mc * (hist_p / cur_p)
            market_caps.append(max(mc, 1.0))
        return np.array(market_caps, dtype=float)

    def run_backtest(self,
                    strategy_name: str,
                    weight_function: Callable,
                    rebalance_dates: Optional[List[pd.Timestamp]] = None) -> Dict:
        """
        Run backtest for a strategy
        
        Parameters:
        -----------
        strategy_name : str
            Name of strategy
        weight_function : Callable
            Function that returns portfolio weights given historical data
        rebalance_dates : List[pd.Timestamp], optional
            Rebalancing dates (if None, use config frequency)
        
        Returns:
        --------
        Dict : Backtest results
        """
        logger.info(f"Running backtest for {strategy_name}")
        
        # Get rebalance dates
        if rebalance_dates is None:
            # Use config start/end to bound rebalance dates,
            # but ensure we have enough lookback history before the first rebalance
            config_start = pd.Timestamp(self.config['backtest']['start_date'])
            config_end = pd.Timestamp(self.config['backtest']['end_date'])
            lookback_start = self.returns.index[self.lookback_period]
            start_date = max(config_start, lookback_start)
            end_date = min(config_end, self.returns.index[-1])
            rebalance_dates = get_rebalance_dates(
                start_date, end_date, self.rebalance_frequency
            )
        
        # Map each rebalance date to the nearest available trading day
        # (handles holidays / weekends — e.g. Labor Day, New Year's)
        trading_days = self.returns.index
        valid_dates = []
        seen = set()
        for d in rebalance_dates:
            if d in trading_days:
                candidate = d
            else:
                future = trading_days[trading_days >= d]
                if len(future) == 0:
                    continue
                candidate = future[0]
            if candidate not in seen:
                valid_dates.append(candidate)
                seen.add(candidate)
        rebalance_dates = valid_dates
        
        logger.info(f"Rebalancing {len(rebalance_dates)} times from {rebalance_dates[0]} to {rebalance_dates[-1]}")
        
        # Initialize tracking variables
        portfolio_values = []
        weights_history = []
        returns_series = []
        turnover_series = []
        results_dates = []
        
        current_weights = np.zeros(len(self.returns.columns))
        portfolio_value = self.initial_capital
        
        # Iterate through rebalance dates
        for i, rebal_date in enumerate(tqdm(rebalance_dates, desc=strategy_name)):
            # Get historical data up to (but not including) rebalance date
            # This ensures no lookahead bias
            hist_returns = ensure_no_lookahead(
                self.returns,
                rebal_date,
                self.lookback_period
            )
            
            if len(hist_returns) < 60:  # Need minimum history
                logger.warning(f"Insufficient history at {rebal_date}, skipping")
                continue
            
            try:
                # Calculate new weights (strategy-specific)
                new_weights = weight_function(hist_returns, rebal_date)
                
                # Apply position limits
                new_weights = apply_position_limits(
                    new_weights,
                    self.min_weight,
                    self.max_weight
                )
                
            except Exception as e:
                logger.error(f"Error calculating weights at {rebal_date}: {e}")
                new_weights = current_weights  # Keep current weights
            
            # Calculate turnover
            turnover = calculate_turnover(current_weights, new_weights)
            
            # Apply transaction costs
            transaction_costs = turnover * (self.transaction_cost + self.slippage) * portfolio_value
            portfolio_value -= transaction_costs
            
            # Update weights
            current_weights = new_weights
            
            # Calculate returns until next rebalance (or end of data)
            if i < len(rebalance_dates) - 1:
                next_rebal = rebalance_dates[i + 1]
            else:
                next_rebal = self.returns.index[-1]
            
            # Get returns between rebalances
            period_returns = self.returns[rebal_date:next_rebal].iloc[1:]  # Exclude rebalance date
            
            for date, row in period_returns.iterrows():
                # Daily portfolio return
                daily_return = (current_weights * row.values).sum()
                portfolio_value *= (1 + daily_return)
                
                results_dates.append(date)
                portfolio_values.append(portfolio_value)
                returns_series.append(daily_return)
                weights_history.append(current_weights.copy())
                turnover_series.append(0)  # No turnover on non-rebalance days
            
            # Record turnover on rebalance day
            if len(turnover_series) > 0:
                turnover_series[-1] = turnover
        
        # Create results dataframe — use actual dates from simulation loop
        results_df = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'returns': returns_series,
            'turnover': turnover_series
        }, index=results_dates)

        # Add weights columns
        weights_df = pd.DataFrame(
            weights_history,
            index=results_dates,
            columns=self.returns.columns
        )
        
        results = {
            'strategy_name': strategy_name,
            'portfolio_values': results_df,
            'weights': weights_df,
            'final_value': portfolio_values[-1] if portfolio_values else self.initial_capital,
            'total_return': (portfolio_values[-1] / self.initial_capital - 1) if portfolio_values else 0,
            'avg_turnover': np.mean(turnover_series)
        }
        
        logger.info(f"{strategy_name} - Final Value: ${results['final_value']:,.0f}, Total Return: {results['total_return']:.2%}")
        
        return results
    
    def run_black_litterman_backtest(self,
                                    llm_generator: Optional[LLMViewGenerator] = None,
                                    data_collector: Optional[DataCollector] = None) -> Dict:
        """
        Run Black-Litterman backtest with LLM-generated views
        
        Parameters:
        -----------
        llm_generator : LLMViewGenerator, optional
            LLM view generator (if None, skip LLM views)
        data_collector : DataCollector, optional
            Data collector for context generation
        
        Returns:
        --------
        Dict : Backtest results
        """
        def bl_weight_function(hist_returns: pd.DataFrame, current_date: pd.Timestamp) -> np.ndarray:
            """Calculate BL weights with LLM views"""
            
            # 用股價比例推算該重平衡日期的歷史市值
            market_caps = self._get_historical_market_caps(
                hist_returns.columns.tolist(), current_date
            )
            
            # Generate LLM views if available
            if llm_generator and data_collector:
                try:
                    # Prepare contexts for each ticker
                    # use_news=True: include news from NewsDatabase if available
                    use_news = self.config.get('llm', {}).get('use_news', True)
                    contexts = {}
                    for ticker in hist_returns.columns:
                        price_data = self.full_prices if self.full_prices is not None else self.prices
                        context = data_collector.prepare_llm_context(
                            ticker, current_date, price_data,
                            use_news=use_news
                        )
                        contexts[ticker] = context

                    # Generate views
                    views_df = llm_generator.generate_views_batch(
                        hist_returns.columns.tolist(),
                        contexts
                    )

                    # Convert to BL format
                    P, Q = llm_generator.convert_to_bl_format(
                        views_df, hist_returns.columns.tolist()
                    )

                    # Dynamic Omega: fusion of confidence + news-technical agreement
                    covariance = hist_returns.cov().values
                    fusion_cfg = self.config.get('llm', {}).get('fusion', {})
                    if use_news and fusion_cfg.get('enabled', True):
                        Omega = llm_generator.calculate_dynamic_omega(
                            views_df, covariance, P,
                            tau=self.config['black_litterman']['tau'],
                            agreement_sensitivity=fusion_cfg.get('agreement_sensitivity', 0.5)
                        )
                    else:
                        Omega = llm_generator.calculate_omega(
                            views_df, covariance, P,
                            tau=self.config['black_litterman']['tau']
                        )

                except Exception as e:
                    logger.error(f"Error generating LLM views: {e}")
                    n_assets = len(hist_returns.columns)

                    # 真正的 no-view（Black-Litterman 退化成 equilibrium）
                    P = np.zeros((0, n_assets))
                    Q = np.zeros(0)
                    Omega = np.zeros((0, 0))
                    
            else:
                # No LLM, no views at all
                n_assets = len(hist_returns.columns)
                P = np.zeros((0, n_assets))
                Q = np.zeros(0)
                Omega = np.zeros((0, 0))
            
            # Run BL optimization
            bl_results = self.bl_model.run_bl_optimization(
                hist_returns, P, Q, Omega, market_caps,
                self.min_weight, self.max_weight
            )
            
            return bl_results['weights']
        
        return self.run_backtest('Black-Litterman', bl_weight_function)
    
    def run_markowitz_backtest(self) -> Dict:
        """Run Markowitz mean-variance backtest"""
        
        def markowitz_weight_function(hist_returns: pd.DataFrame, current_date: pd.Timestamp) -> np.ndarray:
            """Calculate Markowitz weights"""
            results = self.baseline.markowitz_mean_variance(
                hist_returns,
                min_weight=self.min_weight,
                max_weight=self.max_weight
            )
            return results['weights']
        
        return self.run_backtest('Markowitz', markowitz_weight_function)
    
    def run_equal_weight_backtest(self) -> Dict:
        """Run equal weight backtest"""
        
        def equal_weight_function(hist_returns: pd.DataFrame, current_date: pd.Timestamp) -> np.ndarray:
            """Calculate equal weights"""
            return self.baseline.equal_weight(len(hist_returns.columns))
        
        return self.run_backtest('Equal Weight', equal_weight_function)
    
    def run_benchmark_backtest(self) -> Dict:
        """Run SPY benchmark backtest"""
        
        logger.info("Running benchmark (SPY) backtest")
        
        # Simple buy and hold — start from config start_date (not lookback start)
        config_start = pd.Timestamp(self.config['backtest']['start_date'])
        lookback_start = self.returns.index[self.lookback_period]
        bench_start = max(config_start, lookback_start)
        returns_series = self.benchmark_returns[self.benchmark_returns.index >= bench_start]
        
        portfolio_values = [self.initial_capital]
        for ret in returns_series:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        portfolio_values = portfolio_values[1:]  # Remove initial value
        
        results_df = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'returns': returns_series.values,
            'turnover': np.zeros(len(returns_series))
        }, index=returns_series.index)
        
        results = {
            'strategy_name': 'SPY Benchmark',
            'portfolio_values': results_df,
            'weights': None,  # N/A for benchmark
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] / self.initial_capital - 1),
            'avg_turnover': 0.0
        }
        
        logger.info(f"SPY Benchmark - Final Value: ${results['final_value']:,.0f}, Total Return: {results['total_return']:.2%}")
        
        return results
    
    def run_all_strategies(self,
                          llm_generator: Optional[LLMViewGenerator] = None,
                          data_collector: Optional[DataCollector] = None) -> Dict[str, Dict]:
        """
        Run all strategies and compare
        
        Parameters:
        -----------
        llm_generator : LLMViewGenerator, optional
            LLM view generator
        data_collector : DataCollector, optional
            Data collector
        
        Returns:
        --------
        Dict[str, Dict] : Results for all strategies
        """
        all_results = {}
        
        # Black-Litterman with LLM views
        logger.info("\n" + "="*60)
        logger.info("Running Black-Litterman Strategy")
        logger.info("="*60)
        all_results['black_litterman'] = self.run_black_litterman_backtest(
            llm_generator, data_collector
        )
        
        # Markowitz
        logger.info("\n" + "="*60)
        logger.info("Running Markowitz Strategy")
        logger.info("="*60)
        all_results['markowitz'] = self.run_markowitz_backtest()
        
        # Equal Weight
        logger.info("\n" + "="*60)
        logger.info("Running Equal Weight Strategy")
        logger.info("="*60)
        all_results['equal_weight'] = self.run_equal_weight_backtest()
        
        # SPY Benchmark
        logger.info("\n" + "="*60)
        logger.info("Running SPY Benchmark")
        logger.info("="*60)
        all_results['spy_benchmark'] = self.run_benchmark_backtest()
        
        return all_results


def main():
    """Example usage"""
    import yaml
    from utils import setup_logging
    
    setup_logging(level="INFO")
    
    # Load config
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Simulate data
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', '2024-12-31', freq='B')
    n_assets = 7
    
    # Generate correlated returns
    mean_returns = np.array([0.0005] * n_assets)  # ~12% annual
    volatilities = np.array([0.020] * n_assets)  # ~30% annual vol
    
    returns = np.random.multivariate_normal(
        mean_returns,
        np.diag(volatilities ** 2),
        size=len(dates)
    )
    
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=[f'Stock_{i}' for i in range(n_assets)]
    )
    
    # Benchmark (SPY)
    benchmark_returns = np.random.normal(0.0004, 0.015, len(dates))  # ~10% annual
    benchmark_prices = pd.Series(
        100 * np.exp(np.cumsum(benchmark_returns)),
        index=dates,
        name='SPY'
    )
    
    # Run backtest
    engine = BacktestEngine(prices, benchmark_prices, config)
    
    # Run all strategies (without LLM for this example)
    results = engine.run_all_strategies()
    
    # Display results
    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)
    
    for strategy_name, result in results.items():
        print(f"\n{strategy_name.upper()}")
        print(f"  Final Value: ${result['final_value']:,.0f}")
        print(f"  Total Return: {result['total_return']:.2%}")
        print(f"  Avg Turnover: {result['avg_turnover']:.2%}")


if __name__ == "__main__":
    main()
