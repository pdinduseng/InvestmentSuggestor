"""Stock recommendation critic - validates analysis quality"""

import json
import re
from typing import Dict, List, Optional


class StockCritic:
    """
    Reviews stock recommendations for MAJOR inconsistencies and inaccuracies.

    Only flags severe issues that warrant investigation:
    - Severe logical contradictions (BUY with clearly bearish reasoning)
    - Extreme cross-channel conflicts (all channels strongly disagree)
    - Wildly unrealistic price targets (>300% moves)
    - Clear factual impossibilities

    Does NOT flag normal market disagreements or minor variations.
    """

    def __init__(self, llm_provider: str = "anthropic", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize stock critic

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

    def review_stocks(self, aggregated_stocks: List[Dict], video_analyses: List[Dict]) -> Dict:
        """
        Review aggregated stocks for MAJOR issues only

        Args:
            aggregated_stocks: List of aggregated stock recommendations
            video_analyses: Original video analyses for context

        Returns:
            Dictionary with approved stocks, flagged stocks, and summary
        """
        print("  🔍 Reviewing recommendations for major inconsistencies...")

        # Prepare data for LLM review
        stocks_summary = self._prepare_stocks_for_review(aggregated_stocks)

        # Get LLM review
        review_result = self._analyze_with_llm(stocks_summary)

        # Extract flagged tickers
        flagged_tickers = {flag['ticker'] for flag in review_result.get('flagged_stocks', [])}
        approved_tickers = [s['ticker'] for s in aggregated_stocks if s['ticker'] not in flagged_tickers]

        print(f"  ✅ Approved: {len(approved_tickers)} stocks")
        print(f"  ⚠️  Flagged: {len(flagged_tickers)} stocks for investigation")

        return {
            'approved_stocks': approved_tickers,
            'flagged_stocks': review_result.get('flagged_stocks', []),
            'summary': review_result.get('summary', ''),
            'total_reviewed': len(aggregated_stocks)
        }

    def _prepare_stocks_for_review(self, aggregated_stocks: List[Dict]) -> str:
        """Prepare stock data in concise format for LLM review"""
        stocks_data = []

        for stock in aggregated_stocks:
            stock_summary = {
                'ticker': stock['ticker'],
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

            stocks_data.append(stock_summary)

        return json.dumps(stocks_data, indent=2)

    def _analyze_with_llm(self, stocks_summary: str) -> Dict:
        """
        Use LLM to identify MAJOR issues only

        Args:
            stocks_summary: JSON string of stock data

        Returns:
            Review results with flagged stocks
        """
        prompt = f"""You are a critical investment analyst reviewing stock recommendations.

Your job is to identify ONLY MAJOR PROBLEMS that need investigation. Be selective - only flag severe issues.

CRITICAL ISSUES TO FLAG:
1. **Severe Logical Contradictions**
   - Action says "buy" but reasoning is clearly bearish ("company is bankrupt", "revenue declining 80%")
   - Action says "sell" but reasoning is strongly bullish
   - NOT minor disagreements - only clear opposites

2. **Extreme Cross-Channel Conflicts**
   - 3+ channels mention same stock with completely opposite actions AND strong confidence
   - Example: 2 channels say "strong buy, 95% confidence" and 2 say "strong sell, 90% confidence"
   - NOT normal disagreement - only extreme conflicts

3. **Wildly Unrealistic Price Targets**
   - Price target implies >300% move with weak/no justification
   - Price targets that are clearly impossible (e.g., "$5000 for a $50 stock by next month")
   - NOT moderate price targets (50-100% moves are normal in growth investing)

4. **Clear Factual Impossibilities**
   - Claims that are obviously false ("Apple is being acquired", "Tesla went bankrupt")
   - Numerical impossibilities ("P/E ratio of -500")

DO NOT FLAG:
- Normal market disagreements (some bullish, some bearish = normal)
- Different confidence levels on same action
- Moderate price targets (even 100-200% if justified)
- Minor reasoning variations
- Single-channel mentions (no way to verify)
- Stocks with 2 channels where 1 says buy, 1 says hold (normal variation)

STOCKS TO REVIEW:
{stocks_summary}

Return JSON with ONLY stocks that have SEVERE issues:

{{
    "flagged_stocks": [
        {{
            "ticker": "EXAMPLE",
            "concern": "Specific, clear description of the MAJOR problem",
            "severity": "high",
            "investigation_needed": true,
            "investigation_focus": "What specific data to check (price, earnings, news, etc.)"
        }}
    ],
    "summary": "Reviewed X stocks. Y approved, Z flagged for severe issues."
}}

Be very selective. Most stocks should pass review unless there's a MAJOR problem.
"""

        try:
            if self.llm_provider == "openai":
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
        return bool(self.api_key and self.client)
