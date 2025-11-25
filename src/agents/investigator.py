"""Market data investigator - validates flagged stocks with real data"""

import json
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class MarketInvestigator:
    """
    Investigates flagged stocks using real market data from Yahoo Finance.

    Fetches:
    - Current price, market cap, P/E ratio
    - Recent price movements
    - Recent news headlines
    - Analyst price targets (if available)

    Uses LLM to analyze if the concern is valid given real market data.
    """

    def __init__(self, llm_provider: str = "anthropic", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize market investigator

        Args:
            llm_provider: 'openai', 'anthropic', or 'gemini'
            api_key: API key for the LLM provider
            model: Optional model name (uses default if not specified)
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model
        self.client = None

        if llm_provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.model = model or "gpt-4o"
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")

        elif llm_provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key)
                self.model = model or "claude-3-5-sonnet-20241022"
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")

        elif llm_provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model or "gemini-2.0-flash")
                self.model = model or "gemini-2.0-flash"
            except ImportError:
                raise ImportError("google-generativeai package not installed")

    def investigate_stock(self, ticker: str, concern: str, investigation_focus: str,
                         stock_data: Dict) -> Dict:
        """
        Investigate a flagged stock with real market data

        Args:
            ticker: Stock ticker symbol
            concern: The concern raised by the critic
            investigation_focus: Specific aspect to investigate
            stock_data: Original stock data from aggregation

        Returns:
            Investigation results with verdict
        """
        print(f"    🔬 Investigating {ticker}...")

        # Fetch market data
        market_data = self._fetch_market_data(ticker)

        if not market_data:
            print(f"    ⚠️  Could not fetch market data for {ticker}")
            return {
                'ticker': ticker,
                'verdict': 'inconclusive',
                'reasoning': 'Market data unavailable',
                'recommendation': 'caution',
                'market_data': None
            }

        # Analyze with LLM
        verdict = self._analyze_with_market_data(
            ticker, concern, investigation_focus, stock_data, market_data
        )

        print(f"    📊 Verdict: {verdict['verdict'].upper()}")

        return verdict

    def _fetch_market_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch real market data from Yahoo Finance

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with market data or None if failed
        """
        try:
            import yfinance as yf
        except ImportError:
            print("    ⚠️  yfinance not installed. Install with: pip install yfinance")
            return None

        try:
            stock = yf.Ticker(ticker)

            # Get basic info
            info = stock.info

            # Get recent price history (3 months)
            history = stock.history(period="3mo")

            # Get recent news
            try:
                news = stock.news[:5]  # Get top 5 news items
            except:
                news = []

            # Calculate key metrics
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price and not history.empty:
                current_price = history['Close'].iloc[-1]

            # 52-week range
            week_52_high = info.get('fiftyTwoWeekHigh')
            week_52_low = info.get('fiftyTwoWeekLow')

            # Calculate recent performance
            if not history.empty and len(history) > 20:
                month_ago_price = history['Close'].iloc[-21]  # ~1 month
                month_performance = ((current_price - month_ago_price) / month_ago_price * 100) if month_ago_price else None
            else:
                month_performance = None

            market_data = {
                'ticker': ticker,
                'current_price': current_price,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'fifty_two_week_high': week_52_high,
                'fifty_two_week_low': week_52_low,
                'analyst_target_price': info.get('targetMeanPrice'),
                'analyst_target_high': info.get('targetHighPrice'),
                'analyst_target_low': info.get('targetLowPrice'),
                'recommendation_mean': info.get('recommendationMean'),  # 1=Strong Buy, 5=Strong Sell
                'month_performance': month_performance,
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'news_headlines': [
                    {
                        'title': item.get('title', ''),
                        'publisher': item.get('publisher', ''),
                        'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d') if item.get('providerPublishTime') else 'Unknown'
                    }
                    for item in news[:3]  # Top 3 news items
                ],
                'fetched_at': datetime.utcnow().isoformat()
            }

            return market_data

        except Exception as e:
            print(f"    ⚠️  Error fetching market data: {str(e)}")
            return None

    def _format_market_data(self, market_data: Dict) -> str:
        """Format market data for LLM analysis"""
        lines = []

        lines.append(f"Ticker: {market_data['ticker']}")
        lines.append(f"Current Price: ${market_data['current_price']:.2f}" if market_data['current_price'] else "Current Price: N/A")

        if market_data.get('market_cap'):
            market_cap_b = market_data['market_cap'] / 1e9
            lines.append(f"Market Cap: ${market_cap_b:.2f}B")

        if market_data.get('pe_ratio'):
            lines.append(f"P/E Ratio: {market_data['pe_ratio']:.2f}")

        if market_data.get('fifty_two_week_high') and market_data.get('fifty_two_week_low'):
            lines.append(f"52-Week Range: ${market_data['fifty_two_week_low']:.2f} - ${market_data['fifty_two_week_high']:.2f}")

        if market_data.get('month_performance') is not None:
            lines.append(f"1-Month Performance: {market_data['month_performance']:+.2f}%")

        if market_data.get('analyst_target_price'):
            lines.append(f"Analyst Avg Target: ${market_data['analyst_target_price']:.2f}")
            if market_data.get('analyst_target_high') and market_data.get('analyst_target_low'):
                lines.append(f"Analyst Target Range: ${market_data['analyst_target_low']:.2f} - ${market_data['analyst_target_high']:.2f}")

        if market_data.get('recommendation_mean'):
            rec_map = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Sell", 5: "Strong Sell"}
            rec_value = round(market_data['recommendation_mean'])
            lines.append(f"Analyst Consensus: {rec_map.get(rec_value, 'N/A')} ({market_data['recommendation_mean']:.2f})")

        if market_data.get('sector'):
            lines.append(f"Sector: {market_data['sector']}")

        if market_data.get('news_headlines'):
            lines.append("\nRecent News:")
            for news in market_data['news_headlines']:
                lines.append(f"  - [{news['published']}] {news['title']}")

        return "\n".join(lines)

    def _analyze_with_market_data(self, ticker: str, concern: str, investigation_focus: str,
                                  stock_data: Dict, market_data: Dict) -> Dict:
        """
        Use LLM to analyze concern with real market data

        Args:
            ticker: Stock ticker
            concern: The concern raised
            investigation_focus: What to focus on
            stock_data: Original recommendation data
            market_data: Real market data from Yahoo Finance

        Returns:
            Verdict dictionary
        """
        # Format market data
        market_data_str = self._format_market_data(market_data)

        # Format stock recommendation data
        mentions_str = "\n".join([
            f"  - {m['channel']}: {m['action'].upper()} (confidence: {m['confidence']:.0%})\n"
            f"    Reasoning: {m['reasoning'][:150]}\n"
            f"    Price Target: {m.get('price_target', 'Not specified')}"
            for m in stock_data['mentions']
        ])

        prompt = f"""You are investigating a flagged stock recommendation using real market data.

STOCK: {ticker}

CONCERN RAISED:
{concern}

INVESTIGATION FOCUS:
{investigation_focus}

ORIGINAL RECOMMENDATIONS:
{mentions_str}

REAL MARKET DATA:
{market_data_str}

Your task:
1. Compare the YouTuber recommendations to actual market data
2. Check if the concern is valid or can be dismissed
3. Provide a clear verdict

Verdicts:
- **concern_valid**: The concern is real - recommendation has major issues
- **concern_dismissed**: The concern is not valid - recommendation looks reasonable
- **inconclusive**: Not enough data to determine

Return JSON:
{{
    "ticker": "{ticker}",
    "verdict": "concern_valid" | "concern_dismissed" | "inconclusive",
    "reasoning": "Clear explanation comparing recommendations to market data",
    "recommendation": "trust" | "caution" | "reject",
    "key_findings": ["Finding 1", "Finding 2"]
}}

Examples:
- If price target is $500 but current is $100 and analyst targets are $120-150, verdict = concern_valid
- If recommendation is BUY but recent news shows bankruptcy, verdict = concern_valid
- If channels disagree but both have valid points given market data, verdict = concern_dismissed
"""

        try:
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                result = json.loads(response.choices[0].message.content)

            elif self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text
                text = re.sub(r'```json\s*|\s*```', '', text).strip()
                result = json.loads(text)

            elif self.llm_provider == "gemini":
                response = self.client.generate_content(prompt)
                text = response.text
                text = re.sub(r'```json\s*|\s*```', '', text).strip()
                result = json.loads(text)

            # Add market data to result
            result['market_data'] = market_data

            return result

        except Exception as e:
            print(f"    ⚠️  Analysis failed: {str(e)}")
            return {
                'ticker': ticker,
                'verdict': 'inconclusive',
                'reasoning': f'Analysis failed: {str(e)}',
                'recommendation': 'caution',
                'market_data': market_data
            }

    def is_available(self) -> bool:
        """Check if investigator is available"""
        try:
            import yfinance
            return bool(self.api_key and self.client)
        except ImportError:
            return False
