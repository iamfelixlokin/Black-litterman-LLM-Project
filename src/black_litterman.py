"""
Black-Litterman Model Implementation
Combines market equilibrium with subjective views
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class BlackLittermanModel:
    """
    Black-Litterman portfolio optimization model
    """
    
    def __init__(self,
                 risk_aversion: float = 2.5,
                 tau: float = 0.025):
        """
        Initialize Black-Litterman model
        
        Parameters:
        -----------
        risk_aversion : float
            Market risk aversion coefficient (delta)
        tau : float
            Uncertainty in prior (typically 0.01 to 0.05)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        
    def calculate_market_implied_returns(self,
                                        market_caps: np.ndarray,
                                        covariance: np.ndarray) -> np.ndarray:
        """
        Calculate implied equilibrium returns using reverse optimization
        
        Parameters:
        -----------
        market_caps : np.ndarray
            Market capitalizations
        covariance : np.ndarray
            Asset covariance matrix
        
        Returns:
        --------
        np.ndarray : Implied equilibrium returns
        """
        # Market weights (proportional to market cap)
        w_market = market_caps / market_caps.sum()
        
        # Implied returns: Pi = delta * Sigma * w_market
        implied_returns = self.risk_aversion * covariance @ w_market
        
        logger.info(f"Market implied returns: {implied_returns}")
        
        return implied_returns
    
    def calculate_posterior_returns(self,
                                   implied_returns: np.ndarray,
                                   covariance: np.ndarray,
                                   P: np.ndarray,
                                   Q: np.ndarray,
                                   Omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Black-Litterman posterior returns
        
        Parameters:
        -----------
        implied_returns : np.ndarray
            Market implied equilibrium returns
        covariance : np.ndarray
            Asset covariance matrix
        P : np.ndarray
            View matrix (k views x n assets)
        Q : np.ndarray
            View returns vector (k views)
        Omega : np.ndarray
            View uncertainty matrix (k x k diagonal)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (posterior_returns, posterior_covariance)
        """
        # Tau * Sigma (scaled prior covariance)
        tau_sigma = self.tau * covariance
        
        # Posterior returns: E[R] = Pi + tau*Sigma*P'*[P*tau*Sigma*P' + Omega]^-1 * (Q - P*Pi)
        middle_term = P @ tau_sigma @ P.T + Omega
        middle_inv = np.linalg.inv(middle_term)
        
        adjustment = tau_sigma @ P.T @ middle_inv @ (Q - P @ implied_returns)
        posterior_returns = implied_returns + adjustment
        
        # Posterior covariance: Sigma_post = Sigma + tau*Sigma - tau*Sigma*P'*[P*tau*Sigma*P' + Omega]^-1*P*tau*Sigma
        posterior_covariance = covariance + tau_sigma - tau_sigma @ P.T @ middle_inv @ P @ tau_sigma
        
        logger.info(f"Posterior returns: {posterior_returns}")
        
        return posterior_returns, posterior_covariance
    
    def optimize_portfolio(self,
                          expected_returns: np.ndarray,
                          covariance: np.ndarray,
                          min_weight: float = 0.05,
                          max_weight: float = 0.30) -> np.ndarray:
        """
        Optimize portfolio weights using mean-variance optimization
        
        Parameters:
        -----------
        expected_returns : np.ndarray
            Expected returns
        covariance : np.ndarray
            Covariance matrix
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        
        Returns:
        --------
        np.ndarray : Optimal portfolio weights
        """
        n_assets = len(expected_returns)
        
        # Objective: maximize utility = w'*mu - (delta/2)*w'*Sigma*w
        # Equivalent to minimizing: -(w'*mu - (delta/2)*w'*Sigma*w)
        def objective(w):
            portfolio_return = w @ expected_returns
            portfolio_variance = w @ covariance @ w
            # Negative utility (for minimization)
            return -(portfolio_return - (self.risk_aversion / 2) * portfolio_variance)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess (equal weight)
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        weights = result.x
        
        # Ensure weights sum to 1 (numerical precision)
        weights = weights / weights.sum()
        
        return weights
    
    def run_bl_optimization(self,
                          returns: pd.DataFrame,
                          P: np.ndarray,
                          Q: np.ndarray,
                          Omega: np.ndarray,
                          market_caps: Optional[np.ndarray] = None,
                          min_weight: float = 0.05,
                          max_weight: float = 0.30) -> Dict:
        """
        Complete Black-Litterman optimization workflow
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns
        P : np.ndarray
            View matrix
        Q : np.ndarray
            View returns
        Omega : np.ndarray
            View uncertainty matrix
        market_caps : np.ndarray, optional
            Market capitalizations (if None, use equal weight)
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        
        Returns:
        --------
        Dict : Results including weights, returns, statistics
        """
        # Calculate covariance matrix
        covariance = returns.cov().values
        
        # Market implied returns
        if market_caps is None:
            # Use equal weight as market portfolio
            market_caps = np.ones(len(returns.columns))
        
        implied_returns = self.calculate_market_implied_returns(market_caps, covariance)
        
        # Posterior returns with views
        posterior_returns, posterior_cov = self.calculate_posterior_returns(
            implied_returns, covariance, P, Q, Omega
        )
        
        # Optimize portfolio
        weights = self.optimize_portfolio(
            posterior_returns,
            posterior_cov,
            min_weight,
            max_weight
        )
        
        # Calculate portfolio statistics
        portfolio_return = weights @ posterior_returns
        portfolio_volatility = np.sqrt(weights @ posterior_cov @ weights)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        results = {
            'weights': weights,
            'asset_names': returns.columns.tolist(),
            'implied_returns': implied_returns,
            'posterior_returns': posterior_returns,
            'posterior_covariance': posterior_cov,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
        
        logger.info(f"BL Portfolio - Return: {portfolio_return:.4f}, Vol: {portfolio_volatility:.4f}, Sharpe: {sharpe_ratio:.4f}")
        
        return results
    
    def create_weights_dataframe(self, weights: np.ndarray, asset_names: List[str]) -> pd.DataFrame:
        """
        Create formatted weights dataframe
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        asset_names : List[str]
            Asset names
        
        Returns:
        --------
        pd.DataFrame : Formatted weights
        """
        weights_df = pd.DataFrame({
            'Asset': asset_names,
            'Weight': weights,
            'Weight_Pct': weights * 100
        })
        
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        return weights_df


def estimate_covariance(returns: pd.DataFrame, 
                       method: str = "sample",
                       shrinkage_target: float = None) -> np.ndarray:
    """
    Estimate covariance matrix with optional shrinkage
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return data
    method : str
        'sample', 'shrinkage', 'exponential'
    shrinkage_target : float
        Shrinkage intensity (0 to 1)
    
    Returns:
    --------
    np.ndarray : Covariance matrix
    """
    if method == "sample":
        return returns.cov().values
    
    elif method == "shrinkage":
        # Ledoit-Wolf shrinkage
        sample_cov = returns.cov().values
        
        # Target: constant correlation
        n = len(sample_cov)
        avg_corr = (sample_cov.sum() - np.trace(sample_cov)) / (n * (n - 1))
        std_devs = np.sqrt(np.diag(sample_cov))
        target = avg_corr * np.outer(std_devs, std_devs)
        np.fill_diagonal(target, np.diag(sample_cov))
        
        # Apply shrinkage
        if shrinkage_target is None:
            shrinkage_target = 0.2  # Default
        
        shrunk_cov = shrinkage_target * target + (1 - shrinkage_target) * sample_cov
        
        return shrunk_cov
    
    elif method == "exponential":
        # Exponentially weighted covariance
        return returns.ewm(span=60).cov().iloc[-len(returns.columns):].values
    
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    """Example usage"""
    from utils import setup_logging
    
    setup_logging(level="INFO")
    
    # Simulate data
    np.random.seed(42)
    n_assets = 7
    n_periods = 252
    
    # Generate returns
    returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.02,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Market caps (for implied returns)
    market_caps = np.array([2500, 2000, 1800, 1500, 1200, 800, 700])  # Billions
    
    # Views: Asset 0 outperforms by 2%, Asset 3 underperforms by 1%
    P = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # View on Asset 0
        [0, 0, 0, 1, 0, 0, 0],  # View on Asset 3
    ])
    Q = np.array([0.02, -0.01])  # 2% and -1% expected returns
    
    # View uncertainty (proportional to return variance)
    covariance = returns.cov().values
    tau = 0.025
    Omega = np.diag([
        tau * P[0] @ covariance @ P[0].T,
        tau * P[1] @ covariance @ P[1].T
    ])
    
    # Run Black-Litterman
    bl_model = BlackLittermanModel(risk_aversion=2.5, tau=tau)
    
    results = bl_model.run_bl_optimization(
        returns=returns,
        P=P,
        Q=Q,
        Omega=Omega,
        market_caps=market_caps,
        min_weight=0.05,
        max_weight=0.30
    )
    
    # Display results
    weights_df = bl_model.create_weights_dataframe(
        results['weights'],
        results['asset_names']
    )
    
    print("\n=== Black-Litterman Portfolio Weights ===")
    print(weights_df.to_string(index=False))
    
    print(f"\nPortfolio Return: {results['portfolio_return']:.4f}")
    print(f"Portfolio Volatility: {results['portfolio_volatility']:.4f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")


if __name__ == "__main__":
    main()
