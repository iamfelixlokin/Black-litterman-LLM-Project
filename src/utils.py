"""
Utility functions for Black-Litterman portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None, level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def calculate_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Calculate returns from price data
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with datetime index and assets as columns
    method : str
        'simple' or 'log' returns
    
    Returns:
    --------
    pd.DataFrame : Returns data
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return returns.dropna()


def annualize_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate annualized return and volatility
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly)
    
    Returns:
    --------
    Dict with annualized_return and annualized_volatility
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    if years <= 0:
        return {'annualized_return': 0.0, 'annualized_volatility': 0.0}

    annualized_return = (1 + total_return) ** (1 / years) - 1
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility
    }


def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.04,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float : Sharpe ratio
    """
    metrics = annualize_metrics(returns, periods_per_year)
    excess_return = metrics['annualized_return'] - risk_free_rate
    sharpe = excess_return / metrics['annualized_volatility']
    
    return sharpe


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    
    Returns:
    --------
    Tuple : (max_drawdown, peak_date, trough_date)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = cumulative[:trough_date].idxmax()
    
    return max_dd, peak_date, trough_date


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown)
    """
    metrics = annualize_metrics(returns, periods_per_year)
    max_dd, _, _ = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return np.inf
    
    return metrics['annualized_return'] / abs(max_dd)


def calculate_sortino_ratio(returns: pd.Series,
                            risk_free_rate: float = 0.04,
                            periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation)
    """
    metrics = annualize_metrics(returns, periods_per_year)
    excess_return = metrics['annualized_return'] - risk_free_rate
    
    # Downside deviation
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_std == 0:
        return np.inf
    
    return excess_return / downside_std


def get_rebalance_dates(start_date: pd.Timestamp,
                       end_date: pd.Timestamp,
                       frequency: str = "monthly") -> List[pd.Timestamp]:
    """
    Generate rebalance dates based on frequency
    
    Parameters:
    -----------
    start_date : pd.Timestamp
        Start date
    end_date : pd.Timestamp
        End date
    frequency : str
        'daily', 'weekly', 'monthly', 'quarterly'
    
    Returns:
    --------
    List[pd.Timestamp] : List of rebalance dates
    """
    if frequency == "daily":
        dates = pd.date_range(start_date, end_date, freq='B')  # Business days
    elif frequency == "weekly":
        dates = pd.date_range(start_date, end_date, freq='W-FRI')
    elif frequency == "monthly":
        dates = pd.date_range(start_date, end_date, freq='BMS')  # First business day of each month
    elif frequency == "quarterly":
        dates = pd.date_range(start_date, end_date, freq='BQS')  # First business day of each quarter
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    
    return dates.tolist()


def calculate_turnover(old_weights: np.ndarray, 
                       new_weights: np.ndarray) -> float:
    """
    Calculate portfolio turnover
    
    Parameters:
    -----------
    old_weights : np.ndarray
        Old portfolio weights
    new_weights : np.ndarray
        New portfolio weights
    
    Returns:
    --------
    float : Turnover (sum of absolute weight changes)
    """
    return np.abs(new_weights - old_weights).sum()


def apply_position_limits(weights: np.ndarray,
                         min_weight: float = 0.0,
                         max_weight: float = 1.0) -> np.ndarray:
    """
    Apply position size limits to portfolio weights
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    min_weight : float
        Minimum weight per asset
    max_weight : float
        Maximum weight per asset
    
    Returns:
    --------
    np.ndarray : Adjusted weights
    """
    # Clip weights
    clipped = np.clip(weights, min_weight, max_weight)
    
    # Renormalize to sum to 1
    if clipped.sum() > 0:
        clipped = clipped / clipped.sum()
    else:
        # If all weights clipped to 0, use equal weight
        clipped = np.ones_like(weights) / len(weights)
    
    return clipped


def ensure_no_lookahead(data: pd.DataFrame,
                       current_date: pd.Timestamp,
                       lookback_periods: int = None) -> pd.DataFrame:
    """
    Ensure no lookahead bias by filtering data up to current date
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full dataset
    current_date : pd.Timestamp
        Current date for backtesting
    lookback_periods : int, optional
        Number of periods to look back (if None, use all history)
    
    Returns:
    --------
    pd.DataFrame : Filtered data
    """
    # Filter up to (but not including) current date
    historical_data = data[data.index < current_date]
    
    if lookback_periods is not None:
        historical_data = historical_data.tail(lookback_periods)
    
    return historical_data


def calculate_rolling_correlation(returns: pd.DataFrame,
                                  window: int = 60) -> pd.DataFrame:
    """
    Calculate rolling correlation matrix
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return data
    window : int
        Rolling window size
    
    Returns:
    --------
    pd.DataFrame : Latest correlation matrix
    """
    return returns.rolling(window=window).corr().iloc[-len(returns.columns):]


def winsorize_returns(returns: pd.DataFrame,
                     limits: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """
    Winsorize extreme returns to reduce impact of outliers
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return data
    limits : Tuple[float, float]
        Lower and upper percentile limits
    
    Returns:
    --------
    pd.DataFrame : Winsorized returns
    """
    lower = returns.quantile(limits[0])
    upper = returns.quantile(limits[1])
    
    return returns.clip(lower=lower, upper=upper, axis=1)


def format_performance_stats(stats: Dict) -> str:
    """
    Format performance statistics for display
    
    Parameters:
    -----------
    stats : Dict
        Dictionary of performance statistics
    
    Returns:
    --------
    str : Formatted string
    """
    output = []
    output.append("=" * 60)
    output.append("PERFORMANCE STATISTICS")
    output.append("=" * 60)
    
    for key, value in stats.items():
        if isinstance(value, float):
            if 'ratio' in key.lower() or 'return' in key.lower():
                output.append(f"{key:30s}: {value:>10.4f}")
            elif 'drawdown' in key.lower():
                output.append(f"{key:30s}: {value:>10.2%}")
            else:
                output.append(f"{key:30s}: {value:>10.4f}")
        else:
            output.append(f"{key:30s}: {value}")
    
    output.append("=" * 60)
    
    return "\n".join(output)


def save_results(results: Dict, filename: str):
    """
    Save backtest results to file
    
    Parameters:
    -----------
    results : Dict
        Results dictionary
    filename : str
        Output filename
    """
    import pickle
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to {filename}")


def load_results(filename: str) -> Dict:
    """
    Load backtest results from file
    
    Parameters:
    -----------
    filename : str
        Input filename
    
    Returns:
    --------
    Dict : Results dictionary
    """
    import pickle
    
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Results loaded from {filename}")
    return results
