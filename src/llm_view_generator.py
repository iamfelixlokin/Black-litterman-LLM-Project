"""
LLM View Generator for Black-Litterman Model
Uses LLM to analyze unstructured data and generate market views
"""

import anthropic
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMViewGenerator:
    """
    Generates market views using LLM analysis of news, earnings, and macro data
    """
    
    def __init__(self,
                 api_key: str,
                 model: str = "claude-sonnet-4-20250514",
                 temperature: float = 0.3,
                 max_tokens: int = 2000,
                 confidence_omega: Optional[Dict[str, float]] = None):
        """
        Initialize LLM view generator

        Parameters:
        -----------
        api_key : str
            Anthropic API key
        model : str
            Model name
        temperature : float
            Sampling temperature
        max_tokens : int
            Maximum tokens for response
        confidence_omega : dict, optional
            Omega uncertainty scalars per confidence level.
            Defaults to {"high": 0.15, "medium": 0.5, "low": 2.0}
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.confidence_omega = confidence_omega or {
            "high":   0.15,   # Low uncertainty → view影響力大
            "medium": 0.50,   # Medium uncertainty
            "low":    2.00,   # High uncertainty → view影響力小
        }
        
    def create_system_prompt(self) -> str:
        """Create system prompt for Alpha-based view generation with structured-news fusion."""
        return """You are an expert quantitative analyst specializing in relative value investing.

    Your task is to analyze stock performance RELATIVE TO THE MARKET (SPY) and generate Alpha predictions.
    You will receive BOTH structured (price/Alpha) data AND unstructured (news) data.

    CRITICAL DEFINITION:
    Alpha = Stock Return - Market Return

    Example:
    - If stock gains +10% and SPY gains +3%, Alpha = +7% (OUTPERFORMANCE)
    - If stock gains +2% and SPY gains +5%, Alpha = -3% (UNDERPERFORMANCE)

    Your prediction should be the expected ALPHA (excess return vs SPY) over the next 30 days.

    FUSION FRAMEWORK:
    - Structured signals (Alpha trend, momentum) → 70% weight
    - News sentiment → 30% weight
    - When news CONFIRMS the structured signal → increase confidence (higher |alpha|)
    - When news CONTRADICTS the structured signal → be conservative, reduce |alpha|
    - Max news-driven adjustment: ±2% to the base alpha prediction

    Please provide your analysis in the following JSON format:

    {
        "ticker": "STOCK_TICKER",
        "expected_return": X.XX,
        "confidence": "high/medium/low",
        "alpha_trend": "ACCELERATING/STABLE/DECELERATING",
        "news_sentiment_label": "POSITIVE/NEUTRAL/NEGATIVE",
        "fusion_note": "Brief note on how news modified the structured prediction",
        "reasoning": "Brief explanation focusing on RELATIVE performance and fusion logic",
        "key_factors": ["factor1", "factor2", "factor3"],
        "time_horizon": "30 days"
    }

    Guidelines:
    1. expected_return: Alpha prediction (excess return vs SPY in %)

    CRITICAL: This is NOT the absolute return. It is how much the stock will
    OUTPERFORM or UNDERPERFORM the market.

    Analysis framework:
    - Look at historical Alpha trends:
        * 7-day Alpha: +3% → Short-term outperforming
        * 30-day Alpha: +5% → Medium-term outperforming
        * 60-day Alpha: +8% → Long-term outperforming
        → Momentum continuation suggests: Predict +6% to +8% alpha

    - Alpha Trend direction matters:
        * ACCELERATING: Recent alpha improving → boost your prediction
        * DECELERATING: Recent alpha weakening → reduce your prediction
        * STABLE: Consistent relative performance → moderate prediction

    Typical prediction ranges:
    - STRONG OUTPERFORMANCE (30-day alpha > +5%):    → Predict +6% to +10% alpha
    - MODERATE OUTPERFORMANCE (30-day alpha +2-5%):  → Predict +3% to +6% alpha
    - IN LINE WITH MARKET (30-day alpha -2% to +2%): → Predict -1% to +2% alpha
    - MODERATE UNDERPERFORMANCE (-5% to -2%):        → Predict -3% to -6% alpha
    - STRONG UNDERPERFORMANCE (< -5%):               → Predict -6% to -10% alpha

    2. confidence: Based on consistency of Alpha signals AND news alignment
    - "high":   Alpha consistent across 7d/30d/60d AND trend accelerating AND news confirms
    - "medium": Alpha positive but mixed signals, OR trend decelerating, OR news neutral
    - "low":    Alpha trend unstable OR signals conflicting OR news contradicts

    3. alpha_trend: Your reading of the structured technical trend
    - "ACCELERATING" if 7d alpha > 30d alpha (momentum building)
    - "DECELERATING" if 7d alpha < 30d alpha (momentum fading)
    - "STABLE" otherwise

    4. news_sentiment_label: Your reading of the news sentiment
    - "POSITIVE" / "NEUTRAL" / "NEGATIVE"
    - If no news provided, output "NEUTRAL"

    5. fusion_note: How you combined structured and news signals
    - Example: "News confirms bullish alpha trend → raised confidence to high"
    - Example: "News mixed despite strong alpha → kept medium confidence"

    6. reasoning: MUST explain in terms of RELATIVE performance to market
    - Good: "Strong +5% 30-day alpha, ACCELERATING trend, positive news confirms → predict +7% alpha"
    - Bad: "Stock has risen 10% recently"

    7. key_factors: Include "relative_strength", "alpha_momentum", "news_confirmed" etc.

    8. NEVER predict 0% alpha.

    Remember: Structured data = 70% weight, News = 30% weight.
    You are predicting OUTPERFORMANCE vs SPY, not absolute return.

    Think: "Will this stock beat the market by X%?" not "Will this stock go up by X%?"
    """
    
    def generate_view(self, 
                     ticker: str,
                     context: str,
                     market_context: str = "") -> Dict:
        """
        Generate market view for a single asset using LLM
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        context : str
            Formatted context with price, news, fundamentals, macro data
        market_context : str
            Additional market-wide context
        
        Returns:
        --------
        Dict : View with expected_return, confidence, reasoning
        """
        try:
            # Construct user message
            user_message = f"""Please analyze the following information for {ticker} and provide your market view:

{context}

{market_context if market_context else ''}

Provide your analysis in the specified JSON format."""
            
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.create_system_prompt(),
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Parse response
            response_text = message.content[0].text
            view = self._parse_llm_response(response_text, ticker)
            
            logger.info(f"Generated view for {ticker}: {view['expected_return']:.2%} ({view['confidence']})")
            
            return view
            
        except Exception as e:
            logger.error(f"Error generating view for {ticker}: {e}")
            # Return neutral view on error
            return {
                'ticker': ticker,
                'expected_return': 0.0,
                'confidence': 'low',
                'reasoning': f'Error generating view: {str(e)}',
                'key_factors': [],
                'time_horizon': '30 days'
            }
    
    def _parse_llm_response(self, response: str, ticker: str) -> Dict:
        """
        Parse LLM response into structured view
        
        Parameters:
        -----------
        response : str
            Raw LLM response
        ticker : str
            Stock ticker
        
        Returns:
        --------
        Dict : Parsed view
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                view = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Validate and normalize
            view['ticker'] = ticker
            view['expected_return'] = float(view.get('expected_return', 0)) / 100  # Convert % to decimal
            view['confidence'] = view.get('confidence', 'low').lower()
            view['reasoning'] = view.get('reasoning', '')
            view['key_factors'] = view.get('key_factors', [])
            view['time_horizon'] = view.get('time_horizon', '30 days')

            # Fusion fields (new)
            view['alpha_trend'] = view.get('alpha_trend', 'STABLE').upper()
            view['news_sentiment_label'] = view.get('news_sentiment_label', 'NEUTRAL').upper()
            view['fusion_note'] = view.get('fusion_note', '')

            # Sanity check on expected return
            if abs(view['expected_return']) > 0.20:  # More than 20% seems extreme
                logger.warning(f"Extreme return view for {ticker}: {view['expected_return']:.2%}")
                view['expected_return'] = np.clip(view['expected_return'], -0.20, 0.20)

            return view

        except Exception as e:
            logger.error(f"Error parsing response for {ticker}: {e}")
            logger.debug(f"Raw response: {response}")

            # Return neutral view
            return {
                'ticker': ticker,
                'expected_return': 0.0,
                'confidence': 'low',
                'reasoning': 'Failed to parse LLM response',
                'key_factors': [],
                'time_horizon': '30 days',
                'alpha_trend': 'STABLE',
                'news_sentiment_label': 'NEUTRAL',
                'fusion_note': '',
            }
    
    def generate_views_batch(self,
                            tickers: List[str],
                            contexts: Dict[str, str],
                            market_context: str = "") -> pd.DataFrame:
        """
        Generate views for multiple assets
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock tickers
        contexts : Dict[str, str]
            Dictionary mapping ticker to context string
        market_context : str
            Market-wide context
        
        Returns:
        --------
        pd.DataFrame : Views with columns [ticker, expected_return, confidence, reasoning]
        """
        views = []
        
        for ticker in tickers:
            context = contexts.get(ticker, "")
            if not context:
                logger.warning(f"No context available for {ticker}, skipping")
                continue
            
            view = self.generate_view(ticker, context, market_context)
            views.append(view)
        
        views_df = pd.DataFrame(views)
        
        logger.info(f"Generated {len(views_df)} views")
        logger.info(f"Views summary:\n{views_df[['ticker', 'expected_return', 'confidence']]}")
        
        return views_df
    
    def convert_to_bl_format(self, 
                            views_df: pd.DataFrame,
                            tickers: List[str],
                            max_views: int = 7
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert LLM views to Black-Litterman format
        
        Parameters:
        -----------
        views_df : pd.DataFrame
            Views dataframe from generate_views_batch
        tickers : List[str]
            List of tickers in portfolio
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (P matrix, Q vector)
            P: View matrix (n_views x n_assets)
            Q: Expected returns vector (n_views,)
        """
        # 依照 |expected_return| 排序，取最強的幾個
        views_df = views_df.copy()
        views_df['abs_return'] = views_df['expected_return'].abs()
        views_df = views_df.sort_values('abs_return', ascending=False).head(max_views)

        n_assets = len(tickers)
        n_views = len(views_df)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)

        ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}

        for i, row in views_df.iterrows():
            idx = ticker_to_idx[row['ticker']]
            P[i, idx] = 1.0
            Q[i] = row['expected_return']

        return P, Q
    
    def calculate_omega(self,
                       views_df: pd.DataFrame,
                       covariance: np.ndarray,
                       P: np.ndarray,
                       tau: float = 0.025) -> np.ndarray:
        """
        Calculate uncertainty matrix (Omega) from LLM confidence levels only.
        (Structured-data baseline, no news adjustment.)

        Parameters:
        -----------
        views_df : pd.DataFrame
            Views with confidence levels
        covariance : np.ndarray
            Asset covariance matrix
        P : np.ndarray
            View matrix
        tau : float
            Scaling factor for uncertainty

        Returns:
        --------
        np.ndarray : Omega matrix (diagonal)
        """
        n_views = len(views_df)
        confidence_map = self.confidence_omega
        omega = np.zeros(n_views)

        for i, row in views_df.iterrows():
            confidence = row['confidence']
            uncertainty_scalar = confidence_map.get(confidence, 1.0)
            view_variance = tau * (P[i] @ covariance @ P[i].T) * uncertainty_scalar
            omega[i] = view_variance

        return np.diag(omega)

    def calculate_dynamic_omega(self,
                                views_df: pd.DataFrame,
                                covariance: np.ndarray,
                                P: np.ndarray,
                                tau: float = 0.025,
                                agreement_sensitivity: float = 0.5) -> np.ndarray:
        """
        Dynamic Omega via Structured-News Fusion.

        Adjusts uncertainty based on the *agreement* between the structured
        technical signal (Alpha trend) and the news sentiment returned by the LLM.

        Formula
        -------
            base_scalar   = confidence_map[confidence]        (same as calculate_omega)
            agreement     = tech_signal × news_signal ∈ {-1, 0, +1}
            fusion_scalar = base_scalar × exp(-k × agreement)

        where k = agreement_sensitivity (default 0.5).

        Interpretation
        --------------
        agreement = +1 (both bullish or both bearish)
            → fusion_scalar = base × e^{-0.5} ≈ base × 0.61  → LOWER Omega
            → BL places MORE weight on the view (higher conviction)

        agreement = 0 (one signal is neutral)
            → fusion_scalar = base × e^{0} = base              → NO CHANGE

        agreement = -1 (signals conflict)
            → fusion_scalar = base × e^{+0.5} ≈ base × 1.65  → HIGHER Omega
            → BL places LESS weight on the view (lower conviction)

        Parameters
        ----------
        views_df : pd.DataFrame
            Views dataframe; must contain 'confidence', and optionally
            'alpha_trend' (ACCELERATING/STABLE/DECELERATING) and
            'news_sentiment_label' (POSITIVE/NEUTRAL/NEGATIVE).
        covariance : np.ndarray
        P : np.ndarray
        tau : float
        agreement_sensitivity : float
            k in the formula above. Higher → stronger fusion adjustment.

        Returns
        -------
        np.ndarray : Dynamic Omega matrix (diagonal)
        """
        try:
            from news_fetcher import compute_agreement_score
        except ImportError:
            logger.warning("news_fetcher not available – falling back to static Omega")
            return self.calculate_omega(views_df, covariance, P, tau)

        n_views = len(views_df)
        confidence_map = self.confidence_omega
        omega = np.zeros(n_views)

        views_reset = views_df.reset_index(drop=True)

        for i, row in views_reset.iterrows():
            # Base uncertainty from LLM confidence
            confidence = row.get('confidence', 'medium')
            base_scalar = confidence_map.get(confidence, 1.0)

            # Agreement between structured (Alpha trend) and news sentiment
            alpha_trend = row.get('alpha_trend', 'STABLE')
            news_label  = row.get('news_sentiment_label', 'NEUTRAL')
            agreement   = compute_agreement_score(alpha_trend, news_label)

            # Dynamic scaling: higher agreement → lower Omega
            fusion_scalar = base_scalar * np.exp(-agreement_sensitivity * agreement)

            view_variance = tau * (P[i] @ covariance @ P[i].T) * fusion_scalar
            omega[i] = view_variance

            logger.debug(
                f"  [{row.get('ticker', i)}] conf={confidence} "
                f"alpha_trend={alpha_trend} news={news_label} "
                f"agreement={agreement:+.0f} "
                f"base_scalar={base_scalar:.2f} → fusion_scalar={fusion_scalar:.2f}"
            )

        return np.diag(omega)
    
    def summarize_views(self, views_df: pd.DataFrame) -> str:
        """
        Create human-readable summary of views
        
        Parameters:
        -----------
        views_df : pd.DataFrame
            Views dataframe
        
        Returns:
        --------
        str : Formatted summary
        """
        summary = []
        summary.append("=" * 80)
        summary.append("LLM MARKET VIEWS SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        for _, row in views_df.iterrows():
            summary.append(f"Ticker: {row['ticker']}")
            summary.append(f"  Expected Return: {row['expected_return']:>6.2%}")
            summary.append(f"  Confidence: {row['confidence']}")
            summary.append(f"  Reasoning: {row['reasoning']}")
            if row['key_factors']:
                summary.append(f"  Key Factors: {', '.join(row['key_factors'])}")
            summary.append("")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)


def main():
    """Example usage"""
    import os
    from utils import setup_logging
    from data_collection import DataCollector
    
    setup_logging(level="INFO")
    
    # Note: Requires ANTHROPIC_API_KEY environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Initialize
    generator = LLMViewGenerator(api_key=api_key)
    
    # Example context
    context = """=== AAPL Price Performance ===
Current Price: $185.50
30-Day Return: 5.2%
Annualized Volatility: 28.5%

=== Recent News ===
- Apple announces new AI features for iPhone (TechCrunch)
- Strong Q4 earnings beat expectations (Bloomberg)
- iPhone 16 sales exceed forecasts (Reuters)

=== Fundamental Data ===
P/E Ratio: 28.5
Revenue Growth: 8.2%
Profit Margin: 25.3%
Analyst Recommendation: buy

=== Market Environment ===
VIX (Market Volatility): 15.2
10-Year Treasury Yield: 4.25%
"""
    
    # Generate view
    view = generator.generate_view('AAPL', context)
    
    print("\n=== Generated View ===")
    print(json.dumps(view, indent=2))


if __name__ == "__main__":
    main()