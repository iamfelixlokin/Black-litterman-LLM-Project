"""
Baseline portfolio strategies for comparison
Includes: Markowitz, Equal Weight, SPY Benchmark
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaselineStrategies:
    """
    Collection of baseline portfolio strategies
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize baseline strategies
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
    
    def equal_weight(self, n_assets: int) -> np.ndarray:
        """
        Equal weight portfolio (1/N)
        
        Parameters:
        -----------
        n_assets : int
            Number of assets
        
        Returns:
        --------
        np.ndarray : Equal weights
        """
        weights = np.ones(n_assets) / n_assets
        logger.info(f"Equal weight portfolio: {n_assets} assets, {1/n_assets:.4f} each")
        return weights
    
    def markowitz_mean_variance(self,
                               returns: pd.DataFrame,
                               target_return: Optional[float] = None,
                               min_weight: float = 0.0,
                               max_weight: float = 1.0) -> Dict:
        """
        Markowitz mean-variance optimization
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns
        target_return : float, optional
            Target portfolio return (if None, maximize Sharpe ratio)
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        
        Returns:
        --------
        Dict : Optimal weights and statistics
        """
        n_assets = len(returns.columns)
        
        # Calculate expected returns and covariance
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        if target_return is None:
            # Maximize Sharpe ratio
            weights = self._maximize_sharpe(
                mean_returns, cov_matrix, min_weight, max_weight
            )
        else:
            # Minimize variance for target return
            weights = self._minimize_variance(
                mean_returns, cov_matrix, target_return, min_weight, max_weight
            )
        
        # Calculate portfolio statistics
        portfolio_return = weights @ mean_returns
        portfolio_volatility = np.sqrt(weights @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate / 252) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        results = {
            'weights': weights,
            'asset_names': returns.columns.tolist(),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
        
        logger.info(f"Markowitz Portfolio - Return: {portfolio_return:.4f}, Vol: {portfolio_volatility:.4f}, Sharpe: {sharpe_ratio:.4f}")
        
        return results
    
    def _maximize_sharpe(self,
                        mean_returns: np.ndarray,
                        cov_matrix: np.ndarray,
                        min_weight: float,
                        max_weight: float) -> np.ndarray:
        """
        Maximize Sharpe ratio
        """
        n_assets = len(mean_returns)
        
        # Daily risk-free rate
        rf_daily = self.risk_free_rate / 252
        
        def negative_sharpe(w):
            portfolio_return = w @ mean_returns
            portfolio_volatility = np.sqrt(w @ cov_matrix @ w)
            sharpe = (portfolio_return - rf_daily) / portfolio_volatility if portfolio_volatility > 0 else -np.inf
            return -sharpe  # Negative for minimization
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            negative_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            logger.warning(f"Sharpe optimization did not converge: {result.message}")
        
        weights = result.x
        weights = weights / weights.sum()  # Normalize
        
        return weights
    
    def _minimize_variance(self,
                          mean_returns: np.ndarray,
                          cov_matrix: np.ndarray,
                          target_return: float,
                          min_weight: float,
                          max_weight: float) -> np.ndarray:
        """
        Minimize variance for target return
        """
        n_assets = len(mean_returns)
        
        def portfolio_variance(w):
            return w @ cov_matrix @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: w @ mean_returns - target_return}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            logger.warning(f"Variance optimization did not converge: {result.message}")
            # Fall back to max Sharpe
            return self._maximize_sharpe(mean_returns, cov_matrix, min_weight, max_weight)
        
        weights = result.x
        weights = weights / weights.sum()
        
        return weights
    
    def minimum_variance(self,
                        returns: pd.DataFrame,
                        min_weight: float = 0.0,
                        max_weight: float = 1.0) -> Dict:
        """
        Global minimum variance portfolio
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        
        Returns:
        --------
        Dict : Optimal weights and statistics
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov().values
        
        def portfolio_variance(w):
            return w @ cov_matrix @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        weights = result.x / result.x.sum()
        
        mean_returns = returns.mean().values
        portfolio_return = weights @ mean_returns
        portfolio_volatility = np.sqrt(weights @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate / 252) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        results = {
            'weights': weights,
            'asset_names': returns.columns.tolist(),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
        
        logger.info(f"Min Variance Portfolio - Return: {portfolio_return:.4f}, Vol: {portfolio_volatility:.4f}")
        
        return results
    
    def risk_parity(self,
                   returns: pd.DataFrame,
                   min_weight: float = 0.0,
                   max_weight: float = 1.0) -> Dict:
        """
        Risk parity portfolio (equal risk contribution)
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        
        Returns:
        --------
        Dict : Optimal weights and statistics
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov().values
        
        def risk_contribution_error(w):
            # Portfolio volatility
            port_vol = np.sqrt(w @ cov_matrix @ w)
            
            # Marginal contribution to risk
            marginal_contrib = cov_matrix @ w
            
            # Risk contribution
            risk_contrib = w * marginal_contrib / port_vol
            
            # Target: equal risk contribution (1/n each)
            target_rc = np.ones(n_assets) / n_assets
            
            # Sum of squared errors
            return np.sum((risk_contrib - target_rc * port_vol) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            risk_contribution_error,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        weights = result.x / result.x.sum()
        
        mean_returns = returns.mean().values
        portfolio_return = weights @ mean_returns
        portfolio_volatility = np.sqrt(weights @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate / 252) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        results = {
            'weights': weights,
            'asset_names': returns.columns.tolist(),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
        
        logger.info(f"Risk Parity Portfolio - Return: {portfolio_return:.4f}, Vol: {portfolio_volatility:.4f}")
        
        return results


def main():
    """Example usage"""
    from utils import setup_logging
    
    setup_logging(level="INFO")
    
    # Simulate data
    np.random.seed(42)
    n_assets = 7
    n_periods = 252
    
    returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.015 + 0.0004,  # Mean 10% annual
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    baseline = BaselineStrategies(risk_free_rate=0.04)
    
    # Equal weight
    ew_weights = baseline.equal_weight(n_assets)
    print("\n=== Equal Weight ===")
    print(f"Weights: {ew_weights}")
    
    # Markowitz (max Sharpe)
    markowitz = baseline.markowitz_mean_variance(returns, min_weight=0.05, max_weight=0.30)
    print("\n=== Markowitz (Max Sharpe) ===")
    print(f"Weights: {markowitz['weights']}")
    print(f"Sharpe: {markowitz['sharpe_ratio']:.4f}")
    
    # Minimum variance
    min_var = baseline.minimum_variance(returns, min_weight=0.05, max_weight=0.30)
    print("\n=== Minimum Variance ===")
    print(f"Weights: {min_var['weights']}")
    print(f"Volatility: {min_var['portfolio_volatility']:.4f}")
    
    # Risk parity
    risk_par = baseline.risk_parity(returns, min_weight=0.05, max_weight=0.30)
    print("\n=== Risk Parity ===")
    print(f"Weights: {risk_par['weights']}")


if __name__ == "__main__":
    main()
