"""
Data collection module for Black-Litterman portfolio optimization
Collects price data, news, earnings reports, and macro indicators
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import requests
import time

from news_fetcher import NewsDatabase, FinnhubNewsFetcher, aggregate_sentiment

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects financial data for portfolio optimization
    """

    def __init__(self, tickers: List[str], benchmark: str = "SPY",
                 finnhub_api_key: Optional[str] = None):
        """
        Initialize data collector

        Parameters:
        -----------
        tickers : List[str]
            List of stock tickers
        benchmark : str
            Benchmark ticker
        finnhub_api_key : str, optional
            Finnhub API key for live news fetching.
            Falls back to env var FINNHUB_API_KEY if not provided.
        """
        self.tickers = tickers
        self.benchmark = benchmark
        self.all_tickers = tickers + [benchmark]

        # News backend
        self._news_db = NewsDatabase()
        self._news_fetcher = FinnhubNewsFetcher(
            api_key=finnhub_api_key or os.environ.get("FINNHUB_API_KEY", "")
        )
        
    def fetch_price_data(self, 
                        start_date: str, 
                        end_date: str,
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical price data
        
        Parameters:
        -----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        interval : str
            Data interval (1d, 1wk, 1mo)
        
        Returns:
        --------
        pd.DataFrame : Price data (adjusted close)
        """
        logger.info(f"Fetching price data from {start_date} to {end_date}")

        # Use a per-process temp cache dir to avoid SQLite lock conflicts in CI
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), f"yf_cache_{os.getpid()}")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["YFINANCE_CACHE_DIR"] = cache_dir

        try:
            # 批量下載所有股票（更快更穩定）
            data = yf.download(
                self.all_tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                group_by='ticker'
            )
            
            # 提取 Adj Close 價格
            prices = pd.DataFrame()
            
            for ticker in self.all_tickers:
                try:
                    # 處理多層索引的情況
                    if len(self.all_tickers) > 1:
                        if ticker in data.columns.get_level_values(0):
                            ticker_data = data[ticker]
                            if 'Adj Close' in ticker_data.columns:
                                prices[ticker] = ticker_data['Adj Close']
                            elif 'Close' in ticker_data.columns:
                                prices[ticker] = ticker_data['Close']
                    else:
                        # 單一股票的情況
                        if 'Adj Close' in data.columns:
                            prices[ticker] = data['Adj Close']
                        elif 'Close' in data.columns:
                            prices[ticker] = data['Close']
                    
                    logger.info(f"Successfully fetched {ticker} data")
                    
                except Exception as e:
                    logger.warning(f"Error processing {ticker} from batch: {e}, trying individual download")
                    
                    # Fallback: 單獨下載
                    try:
                        single_data = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            interval=interval,
                            progress=False
                        )
                        
                        if not single_data.empty:
                            if 'Adj Close' in single_data.columns:
                                prices[ticker] = single_data['Adj Close']
                            elif 'Close' in single_data.columns:
                                prices[ticker] = single_data['Close']
                            logger.info(f"Successfully fetched {ticker} data (individual)")
                    except Exception as e2:
                        logger.error(f"Failed to fetch {ticker}: {e2}")
            
            # 清理數據
            prices = prices.dropna()
            
            if prices.empty:
                logger.error("No price data was successfully fetched!")
            else:
                logger.info(f"Price data shape: {prices.shape}")
                logger.info(f"Tickers fetched: {list(prices.columns)}")
            
            return prices
            
        except Exception as e:
            logger.error(f"Error in batch download: {e}")
            
            # 完全失敗時嘗試逐個下載
            logger.info("Attempting individual downloads as fallback...")
            prices = pd.DataFrame()
            
            for ticker in self.all_tickers:
                try:
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        progress=False
                    )
                    
                    if not data.empty:
                        if 'Adj Close' in data.columns:
                            prices[ticker] = data['Adj Close']
                        elif 'Close' in data.columns:
                            prices[ticker] = data['Close']
                        logger.info(f"Successfully fetched {ticker} data")
                    
                    time.sleep(0.3)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")
            
            prices = prices.dropna()
            logger.info(f"Fallback fetch complete. Price data shape: {prices.shape}")
            
            return prices
    
    def get_news(self,
                 ticker: str,
                 date: pd.Timestamp,
                 live: bool = False) -> List[Dict]:
        """
        Get news articles for *ticker* up to *date* (no look-ahead bias).

        Strategy
        --------
        1. Check local NewsDatabase (JSON) – used for backtest.
        2. If *live=True* (oracle mode) AND Finnhub key is available, fetch live.
        3. Return empty list if nothing found.

        Parameters
        ----------
        ticker : str
        date : pd.Timestamp
        live : bool
            If True, fall back to Finnhub live fetch when DB has nothing.

        Returns
        -------
        List[Dict] with keys: datetime, headline, summary, source, sentiment
        """
        # 1. Try news database first (backtest / cached data)
        articles = self._news_db.get(ticker, date)

        # 2. Live fallback for oracle
        if not articles and live:
            articles = self._news_fetcher.fetch(ticker, date=date, lookback_days=7)

        logger.debug(f"get_news({ticker}, {date.date()}): {len(articles)} articles")
        return articles

    
    def fetch_earnings_data(self, ticker: str) -> Dict:
        """
        Fetch earnings and fundamental data
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        
        Returns:
        --------
        Dict : Earnings and fundamental data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial data
            info = stock.info
            financials = stock.financials
            earnings = stock.earnings
            
            data = {
                'ticker': ticker,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'profit_margin': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'beta': info.get('beta'),
                'recommendation': info.get('recommendationKey'),
                'target_price': info.get('targetMeanPrice'),
            }
            
            logger.info(f"Fetched fundamental data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching earnings data for {ticker}: {e}")
            return {}
    
    def fetch_macro_indicators(self, date: pd.Timestamp) -> Dict:
        """
        Fetch macroeconomic indicators
        
        Parameters:
        -----------
        date : pd.Timestamp
            Target date
        
        Returns:
        --------
        Dict : Macro indicators (GDP, inflation, rates, etc.)
        """
        macro_data = {
            'date': date,
            'gdp_growth': None,
            'inflation_rate': None,
            'unemployment_rate': None,
            'interest_rate': None,
            'vix': None,
            'treasury_yield_10y': None,
        }
        
        try:
            # Fetch VIX
            vix = yf.download('^VIX', start=date - timedelta(days=5), 
                            end=date, progress=False)
            if not vix.empty:
                # Handle both single and multi-index columns
                if isinstance(vix.columns, pd.MultiIndex):
                    if 'Close' in vix.columns.get_level_values(0):
                        vix_val = vix['Close'].iloc[-1]
                        if isinstance(vix_val, pd.Series):
                            vix_val = vix_val.iloc[0]
                        macro_data['vix'] = float(vix_val) if not pd.isna(vix_val) else None
                elif 'Close' in vix.columns:
                    vix_val = vix['Close'].iloc[-1]
                    macro_data['vix'] = float(vix_val) if not pd.isna(vix_val) else None
            
            # Fetch 10-year treasury
            treasury = yf.download('^TNX', start=date - timedelta(days=5),
                                  end=date, progress=False)
            if not treasury.empty:
                # Handle both single and multi-index columns
                if isinstance(treasury.columns, pd.MultiIndex):
                    if 'Close' in treasury.columns.get_level_values(0):
                        treasury_val = treasury['Close'].iloc[-1]
                        if isinstance(treasury_val, pd.Series):
                            treasury_val = treasury_val.iloc[0]
                        macro_data['treasury_yield_10y'] = float(treasury_val) if not pd.isna(treasury_val) else None
                elif 'Close' in treasury.columns:
                    treasury_val = treasury['Close'].iloc[-1]
                    macro_data['treasury_yield_10y'] = float(treasury_val) if not pd.isna(treasury_val) else None
                
        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
        
        return macro_data
    
    def prepare_llm_context(self,
                       ticker: str,
                       date: pd.Timestamp,
                       price_history: pd.DataFrame,
                       lookback_days: int = 60,
                       use_news: bool = False) -> str:
        """
        Prepare context string for LLM analysis with RELATIVE performance (Alpha)
    
        CRITICAL: Only uses data available up to 'date' to avoid look-ahead bias
    
        Parameters:
        -----------
        ticker : str
            Stock ticker
        date : pd.Timestamp
            Current date (all data must be before this)
        price_history : pd.DataFrame
            Historical price data (must include 'SPY' column for market benchmark)
        lookback_days : int
            Days of history to include
        use_news : bool
            Whether to include historical news (requires pre-downloaded data)
        
        Returns:
        --------
        str : Formatted context for LLM with relative performance metrics
        """
        context_parts = []
    
        # 1. Price Performance (ONLY historical data up to 'date')
        start_date = date - timedelta(days=lookback_days)
        recent_prices = price_history[ticker][start_date:date]
        
        # Check if SPY data is available
        has_spy = 'SPY' in price_history.columns
        if has_spy:
            spy_prices = price_history['SPY'][start_date:date]
        
        if len(recent_prices) > 5:  # Need at least 5 days
            # Calculate returns over different periods
            current_price = float(recent_prices.iloc[-1])
            
            # 7-day returns (asset and market)
            if len(recent_prices) >= 7:
                price_7d_ago = float(recent_prices.iloc[-7])
                return_7d = (current_price / price_7d_ago - 1) * 100
                
                if has_spy:
                    spy_7d_ago = float(spy_prices.iloc[-7])
                    spy_return_7d = (float(spy_prices.iloc[-1]) / spy_7d_ago - 1) * 100
                    alpha_7d = return_7d - spy_return_7d
                else:
                    alpha_7d = None
            else:
                return_7d = 0.0
                alpha_7d = None
            
            # 30-day returns (asset and market)
            if len(recent_prices) >= 30:
                price_30d_ago = float(recent_prices.iloc[-30])
                return_30d = (current_price / price_30d_ago - 1) * 100
                
                if has_spy:
                    spy_30d_ago = float(spy_prices.iloc[-30])
                    spy_return_30d = (float(spy_prices.iloc[-1]) / spy_30d_ago - 1) * 100
                    alpha_30d = return_30d - spy_return_30d
                else:
                    alpha_30d = None
            else:
                return_30d = 0.0
                alpha_30d = None
            
            # 60-day returns (asset and market)
            price_start = float(recent_prices.iloc[0])
            return_full = (current_price / price_start - 1) * 100
            
            if has_spy:
                spy_start = float(spy_prices.iloc[0])
                spy_return_full = (float(spy_prices.iloc[-1]) / spy_start - 1) * 100
                alpha_full = return_full - spy_return_full
            else:
                alpha_full = None
            
            # Volatility
            returns = recent_prices.pct_change().dropna()
            if len(returns) > 1:
                volatility = float(returns.std() * np.sqrt(252) * 100)
            else:
                volatility = 0.0
            
            # Trend strength (regression slope)
            if len(recent_prices) >= 20:
                x = np.arange(len(recent_prices))
                y = np.log(recent_prices.values)
                slope = np.polyfit(x, y, 1)[0]
                trend_strength = slope * 252 * 100  # Annualized
            else:
                trend_strength = 0.0
            
            # Format context
            context_parts.append(f"=== {ticker} Relative Performance Analysis (as of {date.strftime('%Y-%m-%d')}) ===")
            context_parts.append(f"Current Price: ${current_price:.2f}")
            context_parts.append("")
            
            # Absolute returns
            context_parts.append("--- Absolute Returns ---")
            context_parts.append(f"7-Day Return: {return_7d:.2f}%")
            context_parts.append(f"30-Day Return: {return_30d:.2f}%")
            context_parts.append(f"{lookback_days}-Day Return: {return_full:.2f}%")
            context_parts.append(f"Annualized Volatility: {volatility:.2f}%")
            context_parts.append(f"Trend Strength (annualized): {trend_strength:.2f}%")
            context_parts.append("")
            
            # Relative performance (Alpha) - CRITICAL FOR LLM
            if has_spy and alpha_30d is not None:
                context_parts.append("--- RELATIVE Performance vs SPY (Alpha) ---")
                context_parts.append(f"7-Day Alpha: {alpha_7d:+.2f}% (Asset: {return_7d:.2f}% vs SPY: {spy_return_7d:.2f}%)")
                context_parts.append(f"30-Day Alpha: {alpha_30d:+.2f}% (Asset: {return_30d:.2f}% vs SPY: {spy_return_30d:.2f}%)")
                context_parts.append(f"{lookback_days}-Day Alpha: {alpha_full:+.2f}% (Asset: {return_full:.2f}% vs SPY: {spy_return_full:.2f}%)")
                context_parts.append("")
                
                # Relative strength classification
                if alpha_30d > 5:
                    rel_strength = "STRONG OUTPERFORMANCE"
                elif alpha_30d > 2:
                    rel_strength = "MODERATE OUTPERFORMANCE"
                elif alpha_30d > -2:
                    rel_strength = "IN LINE WITH MARKET"
                elif alpha_30d > -5:
                    rel_strength = "MODERATE UNDERPERFORMANCE"
                else:
                    rel_strength = "STRONG UNDERPERFORMANCE"
                
                context_parts.append(f"Relative Strength: {rel_strength}")
                
                # Alpha trend (is relative performance improving?)
                if alpha_7d is not None and alpha_30d is not None:
                    if alpha_7d > alpha_30d + 1:
                        alpha_trend = "ACCELERATING (recent alpha > longer-term alpha)"
                    elif alpha_7d < alpha_30d - 1:
                        alpha_trend = "DECELERATING (recent alpha < longer-term alpha)"
                    else:
                        alpha_trend = "STABLE (consistent relative performance)"
                    
                    context_parts.append(f"Alpha Trend: {alpha_trend}")
                context_parts.append("")
            else:
                context_parts.append("--- Note: SPY data not available, showing absolute returns only ---")
                context_parts.append("")
            
            # Absolute momentum classification (fallback if no SPY)
            if alpha_30d is None:
                if return_30d > 10:
                    momentum = "STRONG UPTREND"
                elif return_30d > 3:
                    momentum = "MODERATE UPTREND"
                elif return_30d > -3:
                    momentum = "SIDEWAYS"
                elif return_30d > -10:
                    momentum = "MODERATE DOWNTREND"
                else:
                    momentum = "STRONG DOWNTREND"
                
                context_parts.append(f"Momentum Classification: {momentum}")
                context_parts.append(f"Recent Acceleration: {'ACCELERATING' if return_7d > return_30d/4 else 'DECELERATING'}")
                context_parts.append("")
        
        # 2. News section (fusion: unstructured data)
        if use_news:
            try:
                news_list = self.get_news(ticker, date)
                if news_list:
                    agg = aggregate_sentiment(news_list)

                    context_parts.append("")
                    context_parts.append("=" * 80)
                    context_parts.append("SUPPLEMENTARY INFORMATION: Recent News (Unstructured Data)")
                    context_parts.append("=" * 80)
                    context_parts.append("")
                    context_parts.append(
                        f"Overall News Sentiment: {agg['label']} "
                        f"(VADER score: {agg['score']:+.2f}, n={agg['n_articles']} articles)"
                    )
                    context_parts.append("")
                    context_parts.append("FUSION ANALYSIS INSTRUCTIONS:")
                    context_parts.append("  - Structured (price/Alpha) signals carry 70% weight")
                    context_parts.append("  - News sentiment carries 30% weight")
                    context_parts.append("  - If news CONFIRMS Alpha trend → increase confidence")
                    context_parts.append("  - If news CONTRADICTS Alpha trend → stay conservative, follow technicals")
                    context_parts.append("  - Max news adjustment to alpha prediction: ±2%")
                    context_parts.append("")
                    context_parts.append("-" * 80)
                    context_parts.append("")

                    for i, article in enumerate(news_list[:5], 1):
                        ts = article.get("datetime", 0)
                        art_date = (
                            datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                            if ts else "N/A"
                        )
                        sent_score = article.get("sentiment", 0.0)
                        context_parts.append(
                            f"[{i}] {art_date}  {article.get('source', 'Unknown')}"
                            f"  sentiment={sent_score:+.2f}"
                        )
                        context_parts.append(f"    Headline: {article.get('headline', 'N/A')}")
                        summary = article.get("summary", "")
                        if summary and summary != article.get("headline", ""):
                            context_parts.append(f"    Summary:  {summary[:120]}")
                        context_parts.append("")

                    context_parts.append("-" * 80)
                    context_parts.append("REMEMBER: Base prediction on Alpha trends FIRST,")
                    context_parts.append("then apply small news adjustment if sentiment is clear & unambiguous.")
                    context_parts.append("=" * 80)
                    context_parts.append("")
            except Exception as e:
                logger.debug(f"Error loading news for {ticker}: {e}")

        return "\n".join(context_parts)
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save data to CSV"""
        data.to_csv(filename)
        logger.info(f"Data saved to {filename}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV"""
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        logger.info(f"Data loaded from {filename}")
        return data


def main():
    """Example usage"""
    from utils import setup_logging
    setup_logging(level="INFO")
    
    # Magnificent 7 stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
    
    collector = DataCollector(tickers)
    
    # Fetch price data
    prices = collector.fetch_price_data(
        start_date='2021-01-01',
        end_date='2024-12-31'
    )
    
    print(f"\nPrice data shape: {prices.shape}")
    print(f"\nFirst few rows:\n{prices.head()}")
    
    # Save data
    collector.save_data(prices, '../data/prices.csv')
    
    # Prepare LLM context example
    context = collector.prepare_llm_context(
        ticker='AAPL',
        date=pd.Timestamp('2024-01-15'),
        price_history=prices
    )
    
    print(f"\n=== LLM Context Example ===\n")
    print(context)


if __name__ == "__main__":
    main()