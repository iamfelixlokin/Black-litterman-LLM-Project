"""
Test script to verify installation and basic functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import matplotlib
        import seaborn
        print("✓ Core libraries imported successfully")
    except ImportError as e:
        print(f"✗ Error importing core libraries: {e}")
        return False
    
    try:
        import yfinance
        print("✓ yfinance imported successfully")
    except ImportError as e:
        print(f"✗ Error importing yfinance: {e}")
        return False
    
    try:
        from src.utils import calculate_returns, annualize_metrics
        from src.data_collection import DataCollector
        from src.black_litterman import BlackLittermanModel
        from src.baseline_strategies import BaselineStrategies
        from src.backtest_engine import BacktestEngine
        from src.performance_metrics import PerformanceAnalyzer
        print("✓ All custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Error importing custom modules: {e}")
        return False
    
    return True


def test_data_generation():
    """Test synthetic data generation"""
    print("\nTesting data generation...")
    
    try:
        import numpy as np
        import pandas as pd
        from src.utils import calculate_returns
        
        # Generate synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
        n_assets = 5
        
        returns = np.random.randn(len(dates), n_assets) * 0.02 + 0.0004
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        # Test return calculation
        test_returns = calculate_returns(prices)
        
        assert len(test_returns) == len(prices) - 1, "Return calculation failed"
        assert not test_returns.isnull().any().any(), "Returns contain NaN"
        
        print("✓ Data generation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False


def test_black_litterman():
    """Test Black-Litterman model"""
    print("\nTesting Black-Litterman model...")
    
    try:
        import numpy as np
        import pandas as pd
        from src.black_litterman import BlackLittermanModel
        
        # Generate synthetic data
        np.random.seed(42)
        n_assets = 5
        n_periods = 252
        
        returns = pd.DataFrame(
            np.random.randn(n_periods, n_assets) * 0.02,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        # Setup views
        P = np.array([[1, 0, 0, 0, 0]])  # View on Asset 0
        Q = np.array([0.02])  # 2% expected return
        Omega = np.array([[0.001]])
        
        market_caps = np.ones(n_assets)
        
        # Run BL
        bl_model = BlackLittermanModel(risk_aversion=2.5, tau=0.025)
        results = bl_model.run_bl_optimization(
            returns, P, Q, Omega, market_caps
        )
        
        # Validate results
        assert 'weights' in results, "Missing weights in results"
        assert len(results['weights']) == n_assets, "Wrong number of weights"
        assert abs(results['weights'].sum() - 1.0) < 1e-6, "Weights don't sum to 1"
        assert (results['weights'] >= 0).all(), "Negative weights found"
        
        print("✓ Black-Litterman model working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Black-Litterman test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_strategies():
    """Test baseline strategies"""
    print("\nTesting baseline strategies...")
    
    try:
        import numpy as np
        import pandas as pd
        from src.baseline_strategies import BaselineStrategies
        
        # Generate data
        np.random.seed(42)
        n_assets = 5
        n_periods = 252
        
        returns = pd.DataFrame(
            np.random.randn(n_periods, n_assets) * 0.015 + 0.0004,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        baseline = BaselineStrategies(risk_free_rate=0.04)
        
        # Test equal weight
        ew_weights = baseline.equal_weight(n_assets)
        assert len(ew_weights) == n_assets
        assert abs(ew_weights.sum() - 1.0) < 1e-6
        
        # Test Markowitz
        markowitz_results = baseline.markowitz_mean_variance(returns)
        assert 'weights' in markowitz_results
        assert abs(markowitz_results['weights'].sum() - 1.0) < 1e-6
        
        # Test minimum variance
        min_var_results = baseline.minimum_variance(returns)
        assert 'weights' in min_var_results
        
        print("✓ Baseline strategies working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Baseline strategies test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_metrics():
    """Test performance metrics calculation"""
    print("\nTesting performance metrics...")
    
    try:
        import numpy as np
        import pandas as pd
        from src.performance_metrics import PerformanceAnalyzer
        from src.utils import calculate_sharpe_ratio, calculate_max_drawdown
        
        # Generate synthetic returns
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
        returns = pd.Series(
            np.random.normal(0.0005, 0.02, len(dates)),
            index=dates
        )
        
        # Test individual metrics
        sharpe = calculate_sharpe_ratio(returns)
        assert not np.isnan(sharpe), "Sharpe ratio is NaN"
        
        max_dd, peak, trough = calculate_max_drawdown(returns)
        assert max_dd <= 0, "Max drawdown should be negative"
        
        # Test analyzer
        analyzer = PerformanceAnalyzer(risk_free_rate=0.04)
        metrics = analyzer.calculate_metrics(returns)
        
        assert 'annualized_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        print("✓ Performance metrics working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'configs', 'config.yaml'
        )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        assert 'assets' in config, "Missing 'assets' in config"
        assert 'backtest' in config, "Missing 'backtest' in config"
        assert 'black_litterman' in config, "Missing 'black_litterman' in config"
        
        assert 'tickers' in config['assets'], "Missing tickers"
        assert len(config['assets']['tickers']) > 0, "No tickers specified"
        
        print("✓ Configuration loaded successfully")
        print(f"  Tickers: {config['assets']['tickers']}")
        print(f"  Backtest period: {config['backtest']['start_date']} to {config['backtest']['end_date']}")
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("BLACK-LITTERMAN PROJECT - INSTALLATION TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Generation Test", test_data_generation),
        ("Black-Litterman Test", test_black_litterman),
        ("Baseline Strategies Test", test_baseline_strategies),
        ("Performance Metrics Test", test_performance_metrics),
        ("Configuration Test", test_config_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Set up your ANTHROPIC_API_KEY in .env file")
        print("2. Run: cd src && python main.py --no-llm")
        print("3. Check results in ../results/ directory")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- Missing dependencies: pip install -r requirements.txt")
        print("- Wrong working directory: run from tests/ folder")
        return 1


if __name__ == "__main__":
    exit(main())
