"""
Performance metrics and visualization for portfolio backtests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

from utils import (
    annualize_metrics, calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_calmar_ratio, calculate_sortino_ratio
)

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class PerformanceAnalyzer:
    """
    Analyze and visualize portfolio performance
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize performance analyzer
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self,
                         returns: pd.Series,
                         benchmark_returns: pd.Series = None,
                         periods_per_year: int = 252) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series, optional
            Benchmark returns for comparison
        periods_per_year : int
            Periods per year (252 for daily, 12 for monthly)
        
        Returns:
        --------
        Dict : Performance metrics
        """
        metrics = {}
        
        # Basic metrics
        ann_metrics = annualize_metrics(returns, periods_per_year)
        metrics['annualized_return'] = ann_metrics['annualized_return']
        metrics['annualized_volatility'] = ann_metrics['annualized_volatility']
        
        # Risk-adjusted returns
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(
            returns, self.risk_free_rate, periods_per_year
        )
        metrics['sortino_ratio'] = calculate_sortino_ratio(
            returns, self.risk_free_rate, periods_per_year
        )
        
        # Drawdown
        max_dd, peak_date, trough_date = calculate_max_drawdown(returns)
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_peak'] = peak_date
        metrics['max_drawdown_trough'] = trough_date
        
        # Calmar ratio
        metrics['calmar_ratio'] = calculate_calmar_ratio(returns, periods_per_year)
        
        # Win rate
        metrics['win_rate'] = (returns > 0).sum() / len(returns)
        
        # Best/Worst periods
        metrics['best_day'] = returns.max()
        metrics['worst_day'] = returns.min()
        
        # Skewness and Kurtosis
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Value at Risk (95% and 99%)
        metrics['var_95'] = returns.quantile(0.05)
        metrics['var_99'] = returns.quantile(0.01)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # Benchmark comparison
        if benchmark_returns is not None:
            # Align indices
            aligned_returns = returns.align(benchmark_returns, join='inner')
            port_ret = aligned_returns[0]
            bench_ret = aligned_returns[1]
            
            # Beta
            covariance = np.cov(port_ret, bench_ret)[0, 1]
            benchmark_variance = bench_ret.var()
            metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha
            rf_daily = self.risk_free_rate / periods_per_year
            metrics['alpha'] = (metrics['annualized_return'] - 
                              (rf_daily * periods_per_year + metrics['beta'] * 
                               (annualize_metrics(bench_ret, periods_per_year)['annualized_return'] - rf_daily * periods_per_year)))
            
            # Information ratio
            excess_returns = port_ret - bench_ret
            tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
            metrics['information_ratio'] = (excess_returns.mean() * periods_per_year / tracking_error 
                                          if tracking_error > 0 else 0)
            
            # Correlation
            metrics['correlation_with_benchmark'] = port_ret.corr(bench_ret)
        
        return metrics
    
    def create_performance_summary(self,
                                  results: Dict[str, Dict],
                                  benchmark_returns: pd.Series = None) -> pd.DataFrame:
        """
        Create summary table comparing all strategies
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            Results from all strategies
        benchmark_returns : pd.Series, optional
            Benchmark returns
        
        Returns:
        --------
        pd.DataFrame : Summary table
        """
        summary_data = []
        
        for strategy_name, result in results.items():
            returns = result['portfolio_values']['returns']
            
            metrics = self.calculate_metrics(returns, benchmark_returns)
            
            summary_data.append({
                'Strategy': strategy_name,
                'Total Return': result['total_return'],
                'Ann. Return': metrics['annualized_return'],
                'Ann. Volatility': metrics['annualized_volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Sortino Ratio': metrics['sortino_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Calmar Ratio': metrics['calmar_ratio'],
                'Win Rate': metrics['win_rate'],
                'Avg Turnover': result.get('avg_turnover', 0),
            })
            
            if benchmark_returns is not None:
                summary_data[-1].update({
                    'Beta': metrics.get('beta', np.nan),
                    'Alpha': metrics.get('alpha', np.nan),
                    'Info Ratio': metrics.get('information_ratio', np.nan)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df
    
    def plot_cumulative_returns(self,
                               results: Dict[str, Dict],
                               save_path: str = None):
        """
        Plot cumulative returns for all strategies
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            Results from all strategies
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for strategy_name, result in results.items():
            portfolio_values = result['portfolio_values']['portfolio_value']
            cumulative_returns = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
            
            ax.plot(cumulative_returns.index, cumulative_returns.values, 
                   label=strategy_name, linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cumulative returns plot saved to {save_path}")
        
        plt.show()
    
    def plot_drawdowns(self,
                      results: Dict[str, Dict],
                      save_path: str = None):
        """
        Plot drawdown charts for all strategies
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            Results from all strategies
        save_path : str, optional
            Path to save figure
        """
        n_strategies = len(results)
        fig, axes = plt.subplots(n_strategies, 1, figsize=(14, 4 * n_strategies))
        
        if n_strategies == 1:
            axes = [axes]
        
        for ax, (strategy_name, result) in zip(axes, results.items()):
            portfolio_values = result['portfolio_values']['portfolio_value']
            
            # Calculate drawdown
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max * 100
            
            ax.fill_between(drawdown.index, 0, drawdown.values, 
                           alpha=0.3, color='red', label='Drawdown')
            ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
            
            ax.set_ylabel('Drawdown (%)', fontsize=10)
            ax.set_title(f'{strategy_name} - Drawdown', fontsize=12)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.xlabel('Date', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drawdown plots saved to {save_path}")
        
        plt.show()
    
    def plot_rolling_metrics(self,
                            results: Dict[str, Dict],
                            window: int = 60,
                            save_path: str = None):
        """
        Plot rolling Sharpe ratio and volatility
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            Results from all strategies
        window : int
            Rolling window size
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        for strategy_name, result in results.items():
            returns = result['portfolio_values']['returns']
            
            # Rolling Sharpe ratio
            rolling_sharpe = (returns.rolling(window).mean() * 252 - self.risk_free_rate) / \
                           (returns.rolling(window).std() * np.sqrt(252))
            
            ax1.plot(rolling_sharpe.index, rolling_sharpe.values, 
                    label=strategy_name, linewidth=2, alpha=0.7)
            
            # Rolling volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
            
            ax2.plot(rolling_vol.index, rolling_vol.values,
                    label=strategy_name, linewidth=2, alpha=0.7)
        
        ax1.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
        ax1.set_title(f'Rolling Sharpe Ratio ({window}-day window)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Annualized Volatility (%)', fontsize=12)
        ax2.set_title(f'Rolling Volatility ({window}-day window)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Rolling metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_weights_evolution(self,
                              weights: pd.DataFrame,
                              strategy_name: str,
                              save_path: str = None):
        """
        Plot portfolio weights over time
        
        Parameters:
        -----------
        weights : pd.DataFrame
            Portfolio weights over time
        strategy_name : str
            Strategy name
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Resample to reduce clutter (e.g., monthly)
        weights_monthly = weights.resample('M').last()
        
        ax.stackplot(weights_monthly.index, 
                    *[weights_monthly[col].values for col in weights_monthly.columns],
                    labels=weights_monthly.columns,
                    alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Weight', fontsize=12)
        ax.set_title(f'{strategy_name} - Portfolio Weights Evolution', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Weights evolution plot saved to {save_path}")
        
        plt.show()
    
    def plot_return_distribution(self,
                                results: Dict[str, Dict],
                                save_path: str = None):
        """
        Plot return distribution histograms
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            Results from all strategies
        save_path : str, optional
            Path to save figure
        """
        n_strategies = len(results)
        fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 5))
        
        if n_strategies == 1:
            axes = [axes]
        
        for ax, (strategy_name, result) in zip(axes, results.items()):
            returns = result['portfolio_values']['returns'] * 100  # Convert to %
            
            ax.hist(returns.values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(x=returns.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {returns.mean():.2f}%')
            ax.axvline(x=returns.median(), color='green', linestyle='--',
                      linewidth=2, label=f'Median: {returns.median():.2f}%')
            
            ax.set_xlabel('Daily Return (%)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{strategy_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Return distribution plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self,
                       results: Dict[str, Dict],
                       benchmark_returns: pd.Series = None,
                       output_dir: str = "results"):
        """
        Generate comprehensive performance report
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            Results from all strategies
        benchmark_returns : pd.Series, optional
            Benchmark returns
        output_dir : str
            Output directory for plots and tables
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating performance report...")
        
        # Summary table
        summary = self.create_performance_summary(results, benchmark_returns)
        summary_path = f"{output_dir}/performance_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"Summary table saved to {summary_path}")
        
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(summary.to_string(index=False))
        print("="*80)
        
        # Plots
        self.plot_cumulative_returns(results, f"{output_dir}/cumulative_returns.png")
        self.plot_drawdowns(results, f"{output_dir}/drawdowns.png")
        self.plot_rolling_metrics(results, save_path=f"{output_dir}/rolling_metrics.png")
        self.plot_return_distribution(results, f"{output_dir}/return_distributions.png")
        
        # Individual weight plots
        for strategy_name, result in results.items():
            if result['weights'] is not None:
                safe_name = strategy_name.replace(' ', '_').lower()
                self.plot_weights_evolution(
                    result['weights'],
                    strategy_name,
                    f"{output_dir}/weights_{safe_name}.png"
                )
        
        logger.info(f"Report generation complete. Files saved to {output_dir}/")


def main():
    """Example usage"""
    from utils import setup_logging
    import numpy as np
    
    setup_logging(level="INFO")
    
    # Simulate results
    dates = pd.date_range('2021-01-01', '2024-12-31', freq='B')
    
    # Strategy 1: Higher return, higher vol
    returns1 = pd.Series(np.random.normal(0.0006, 0.02, len(dates)), index=dates)
    values1 = pd.DataFrame({
        'portfolio_value': 1000000 * (1 + returns1).cumprod(),
        'returns': returns1,
        'turnover': np.random.uniform(0, 0.1, len(dates))
    }, index=dates)
    
    # Strategy 2: Lower return, lower vol
    returns2 = pd.Series(np.random.normal(0.0004, 0.015, len(dates)), index=dates)
    values2 = pd.DataFrame({
        'portfolio_value': 1000000 * (1 + returns2).cumprod(),
        'returns': returns2,
        'turnover': np.random.uniform(0, 0.05, len(dates))
    }, index=dates)
    
    results = {
        'Strategy A': {
            'portfolio_values': values1,
            'weights': None,
            'final_value': values1['portfolio_value'].iloc[-1],
            'total_return': values1['portfolio_value'].iloc[-1] / 1000000 - 1,
            'avg_turnover': 0.05
        },
        'Strategy B': {
            'portfolio_values': values2,
            'weights': None,
            'final_value': values2['portfolio_value'].iloc[-1],
            'total_return': values2['portfolio_value'].iloc[-1] / 1000000 - 1,
            'avg_turnover': 0.03
        }
    }
    
    # Analyze
    analyzer = PerformanceAnalyzer(risk_free_rate=0.04)
    analyzer.generate_report(results, output_dir="../results")


if __name__ == "__main__":
    main()
