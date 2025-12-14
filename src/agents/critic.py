"""Stock recommendation critic - validates analysis quality"""

import json
import re
from typing import Dict, List, Optional


class StockCritic:
    """
    Provides critical analysis and counterpoints for stock recommendations.

    For each stock recommendation, provides:
    - Critical perspective on the bullish/bearish thesis
    - Potential risks and counterarguments
    - Red flags or concerns investors should consider
    - Alternative viewpoints to help readers see both sides

    Does NOT remove stocks or flag them as invalid.
    Instead, adds balanced critical analysis to help readers make informed decisions.
    """

    def __init__(self, llm_provider: str = "anthropic", api_key: Optional[str] = None, model: Optional[str] = None, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize stock critic

        Args:
            llm_provider: 'openai', 'anthropic', 'gemini', or 'ollama'
            api_key: API key for the LLM provider (not needed for ollama)
            model: Optional model name (uses default if not specified)
            ollama_base_url: Base URL for Ollama server (default: http://localhost:11434)
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.client = None

        if llm_provider == "ollama":
            # Ollama doesn't need a special client, we'll use requests
            self.model = model or "qwen2.5:3b"
            self.client = "ollama"  # Marker to indicate it's available

        elif llm_provider == "openai":
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

    def review_stocks(self, aggregated_stocks: List[Dict], video_analyses: List[Dict], market_data: Optional[Dict] = None) -> Dict:
        """
        Provide critical analysis for each stock recommendation

        Args:
            aggregated_stocks: List of aggregated stock recommendations
            video_analyses: Original video analyses for context
            market_data: Optional dict mapping ticker -> market data from Yahoo Finance

        Returns:
            Dictionary with critical analysis for each stock
        """
        print("  🔍 Adding critical analysis to recommendations...")
        print(f"  Analyzing {len(aggregated_stocks)} stocks individually...")

        critical_analysis_map = {}
        failed_count = 0

        # Loop through each stock for focused analysis
        for i, stock in enumerate(aggregated_stocks, 1):
            ticker = stock['ticker']
            print(f"    [{i}/{len(aggregated_stocks)}] {ticker}...", end=' ')

            try:
                # Prepare single stock data
                stock_data = self._prepare_single_stock_for_review(stock, market_data)

                # Get LLM critical analysis for this stock
                analysis = self._analyze_single_stock_with_llm(
                    stock_data,
                    has_market_data=bool(market_data and ticker in market_data)
                )

                critical_analysis_map[ticker] = {
                    'counterpoints': analysis.get('counterpoints', []),
                    'risks': analysis.get('risks', []),
                    'red_flags': analysis.get('red_flags', []),
                    'alternative_view': analysis.get('alternative_view', '')
                }
                print("✅")

            except Exception as e:
                print(f"❌ {str(e)[:30]}")
                failed_count += 1
                # Continue with other stocks even if one fails

        success_count = len(critical_analysis_map)
        print(f"\n  ✅ Added critical analysis to {success_count}/{len(aggregated_stocks)} stocks")
        if failed_count > 0:
            print(f"  ⚠️  {failed_count} stock(s) failed analysis")

        return {
            'critical_analysis': critical_analysis_map,
            'summary': f'Added critical analysis to {success_count} stocks. {failed_count} failed.' if failed_count > 0 else f'Added critical analysis to {success_count} stocks.',
            'total_reviewed': len(aggregated_stocks)
        }

    def _prepare_single_stock_for_review(self, stock: Dict, market_data: Optional[Dict] = None) -> str:
        """Prepare single stock data for LLM review"""
        ticker = stock['ticker']
        stock_summary = {
            'ticker': ticker,
            'company': stock.get('company_name', 'Unknown'),
            'num_channels': stock['num_channels'],
            'sentiment_aligned': stock['sentiment_aligned'],
            'dominant_action': stock['dominant_action'],
            'avg_confidence': stock['avg_confidence'],
            'mentions': []
        }

        # Add mention details
        for mention in stock['mentions']:
            stock_summary['mentions'].append({
                'channel': mention['channel'],
                'action': mention['action'],
                'reasoning': mention['reasoning'],
                'confidence': mention['confidence'],
                'price_target': mention.get('price_target'),
                'catalysts': mention.get('catalysts', [])
            })

        # Add market data if available
        if market_data and ticker in market_data:
            mdata = market_data[ticker]
            stock_summary['market_data'] = {
                'current_price': mdata.get('current_price'),
                'pe_ratio': mdata.get('pe_ratio'),
                'forward_pe': mdata.get('forward_pe'),
                'analyst_target': mdata.get('analyst_target_price'),
                'analyst_target_range': f"{mdata.get('analyst_target_low')}-{mdata.get('analyst_target_high')}" if mdata.get('analyst_target_low') else None,
                'recommendation_mean': mdata.get('recommendation_mean'),
                'month_performance': mdata.get('month_performance'),
                '52w_range': f"{mdata.get('fifty_two_week_low')}-{mdata.get('fifty_two_week_high')}" if mdata.get('fifty_two_week_low') else None,
                'sector': mdata.get('sector'),
                'recent_news': [n['title'] for n in mdata.get('news_headlines', [])[:2]]
            }

        return json.dumps(stock_summary, indent=2)

    def _prepare_stocks_for_review(self, aggregated_stocks: List[Dict], market_data: Optional[Dict] = None) -> str:
        """Prepare stock data in concise format for LLM review"""
        stocks_data = []

        for stock in aggregated_stocks:
            ticker = stock['ticker']
            stock_summary = {
                'ticker': ticker,
                'company': stock.get('company_name', 'Unknown'),
                'num_channels': stock['num_channels'],
                'sentiment_aligned': stock['sentiment_aligned'],
                'dominant_action': stock['dominant_action'],
                'avg_confidence': stock['avg_confidence'],
                'mentions': []
            }

            # Add mention details
            for mention in stock['mentions']:
                stock_summary['mentions'].append({
                    'channel': mention['channel'],
                    'action': mention['action'],
                    'reasoning': mention['reasoning'][:200],  # Truncate for brevity
                    'confidence': mention['confidence'],
                    'price_target': mention.get('price_target'),
                    'catalysts': mention.get('catalysts', [])
                })

            # Add market data if available
            if market_data and ticker in market_data:
                mdata = market_data[ticker]
                stock_summary['market_data'] = {
                    'current_price': mdata.get('current_price'),
                    'pe_ratio': mdata.get('pe_ratio'),
                    'forward_pe': mdata.get('forward_pe'),
                    'analyst_target': mdata.get('analyst_target_price'),
                    'analyst_target_range': f"{mdata.get('analyst_target_low')}-{mdata.get('analyst_target_high')}" if mdata.get('analyst_target_low') else None,
                    'recommendation_mean': mdata.get('recommendation_mean'),
                    'month_performance': mdata.get('month_performance'),
                    '52w_range': f"{mdata.get('fifty_two_week_low')}-{mdata.get('fifty_two_week_high')}" if mdata.get('fifty_two_week_low') else None,
                    'sector': mdata.get('sector'),
                    'recent_news': [n['title'] for n in mdata.get('news_headlines', [])[:2]]
                }

            stocks_data.append(stock_summary)

        return json.dumps(stocks_data, indent=2)

    def _analyze_single_stock_with_llm(self, stock_data: str, has_market_data: bool = False) -> Dict:
        """
        Use LLM to provide critical analysis for a single stock

        Args:
            stock_data: JSON string of single stock data
            has_market_data: Whether real market data is included

        Returns:
            Critical analysis for the stock
        """
        market_data_instruction = ""
        if has_market_data:
            market_data_instruction = """
**IMPORTANT: Real market data is included for this stock.**
You MUST use this data to validate claims:
- Compare YouTuber price targets to current price and analyst targets
- Check if "undervalued" claims match actual P/E ratios
- Verify if sentiment matches analyst recommendations
- Use recent news to identify risks not mentioned
- Compare performance claims to actual 1-month returns

Be data-driven and specific. Call out discrepancies between claims and reality.
"""

        prompt = f"""You are a critical investment analyst providing balanced counterpoints to a stock recommendation.
{market_data_instruction}

Your job is to help readers see BOTH SIDES of this investment thesis by providing critical analysis, potential risks, and alternative perspectives.

Provide:

1. **Counterpoints** - Arguments against the bullish/bearish thesis
   - What could go wrong with this investment?
   - What assumptions might be too optimistic/pessimistic?
   - What are the opposing views?

2. **Risks** - Specific risks investors should consider
   - Market risks, competitive risks, execution risks
   - Regulatory, economic, or technological risks
   - Company-specific concerns

3. **Red Flags** - Warning signs or concerns (if any)
   - Valuation concerns, deteriorating fundamentals
   - Management issues, competitive threats
   - Only include if legitimately concerning

4. **Alternative View** - A brief alternative perspective
   - How might a bear/bull (opposite of recommendation) view this?
   - What's the contrarian case?

IMPORTANT:
- Be balanced and fair - don't manufacture concerns
- Focus on helping readers make informed decisions
- If channels disagree, explore both perspectives
- Be specific and cite the market data when available

STOCK TO REVIEW:
{stock_data}

Return JSON:

{{
    "ticker": "TICKER",
    "counterpoints": ["Counterpoint 1", "Counterpoint 2", "Counterpoint 3"],
    "risks": ["Risk 1", "Risk 2", "Risk 3"],
    "red_flags": ["Red flag 1 (if any)", "Red flag 2 (if any)"],
    "alternative_view": "Brief contrarian perspective (2-3 sentences)"
}}

Provide thoughtful, balanced critical analysis.
"""

        try:
            if self.llm_provider == "ollama":
                import requests
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.1
                        }
                    },
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                text = result.get('response', '')
                # Remove markdown if present
                text = re.sub(r'```json\s*|\s*```', '', text).strip()
                return json.loads(text)

            elif self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                return json.loads(response.choices[0].message.content)

            elif self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                # Extract JSON from response
                text = response.content[0].text
                # Remove markdown if present
                text = re.sub(r'```json\s*|\s*```', '', text).strip()
                return json.loads(text)

            elif self.llm_provider == "gemini":
                response = self.client.generate_content(prompt)
                text = response.text
                # Remove markdown if present
                text = re.sub(r'```json\s*|\s*```', '', text).strip()
                return json.loads(text)

        except Exception as e:
            raise Exception(f"LLM analysis failed: {str(e)}")

    def _analyze_with_llm(self, stocks_summary: str, has_market_data: bool = False) -> Dict:
        """
        Use LLM to provide critical analysis

        Args:
            stocks_summary: JSON string of stock data
            has_market_data: Whether real market data is included

        Returns:
            Review results with critical analysis
        """
        market_data_instruction = ""
        if has_market_data:
            market_data_instruction = """
**IMPORTANT: Real market data is included for each stock.**
You MUST use this data to validate claims:
- Compare YouTuber price targets to current price and analyst targets
- Check if "undervalued" claims match actual P/E ratios
- Verify if sentiment matches analyst recommendations
- Use recent news to identify risks not mentioned
- Compare performance claims to actual 1-month returns

Be data-driven and specific. Call out discrepancies between claims and reality.
"""

        prompt = f"""You are a critical investment analyst providing balanced counterpoints to stock recommendations.
{market_data_instruction}

Your job is to help readers see BOTH SIDES of investment theses by providing critical analysis, potential risks, and alternative perspectives.

For EACH stock recommendation, provide:

1. **Counterpoints** - Arguments against the bullish/bearish thesis
   - What could go wrong with this investment?
   - What assumptions might be too optimistic?
   - What are the opposing views?

2. **Risks** - Specific risks investors should consider
   - Market risks, competitive risks, execution risks
   - Regulatory, economic, or technological risks
   - Company-specific concerns

3. **Red Flags** - Warning signs or concerns (if any)
   - Valuation concerns, deteriorating fundamentals
   - Management issues, competitive threats
   - Only include if legitimately concerning

4. **Alternative View** - A brief alternative perspective
   - How might a bear/bull (opposite of recommendation) view this?
   - What's the contrarian case?

IMPORTANT:
- Provide critical analysis for ALL stocks, not just problematic ones
- Be balanced and fair - don't manufacture concerns
- Focus on helping readers make informed decisions
- If channels disagree, explore both perspectives

STOCKS TO REVIEW:
{stocks_summary}

Return JSON:

{{
    "stock_analyses": [
        {{
            "ticker": "TICKER",
            "counterpoints": ["Counterpoint 1", "Counterpoint 2"],
            "risks": ["Risk 1", "Risk 2"],
            "red_flags": ["Red flag 1 (if any)"],
            "alternative_view": "Brief contrarian perspective"
        }}
    ],
    "summary": "Added critical analysis to X stocks to provide balanced perspective."
}}

Provide thoughtful, balanced critical analysis for every stock.
"""

        try:
            if self.llm_provider == "ollama":
                import requests
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.1
                        }
                    },
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                text = result.get('response', '')
                # Remove markdown if present
                text = re.sub(r'```json\s*|\s*```', '', text).strip()
                return json.loads(text)

            elif self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                return json.loads(response.choices[0].message.content)

            elif self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                # Extract JSON from response
                text = response.content[0].text
                # Remove markdown if present
                text = re.sub(r'```json\s*|\s*```', '', text).strip()
                return json.loads(text)

            elif self.llm_provider == "gemini":
                response = self.client.generate_content(prompt)
                text = response.text
                # Remove markdown if present
                text = re.sub(r'```json\s*|\s*```', '', text).strip()
                return json.loads(text)

        except Exception as e:
            print(f"  ⚠️ Critic analysis failed: {str(e)}")
            # Return empty result if critic fails
            return {
                'flagged_stocks': [],
                'summary': f'Critic analysis failed: {str(e)}'
            }

    def is_available(self) -> bool:
        """Check if critic is available"""
        if self.llm_provider == "ollama":
            return bool(self.client)  # ollama doesn't need API key
        return bool(self.api_key and self.client)
