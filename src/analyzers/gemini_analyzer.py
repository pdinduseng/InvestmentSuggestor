"""Gemini-based video analyzer"""

import json
import re
from typing import Dict
from .base import VideoAnalyzer, AnalysisError


class GeminiVideoAnalyzer(VideoAnalyzer):
    """Uses Gemini API to analyze YouTube videos directly"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini analyzer

        Args:
            api_key: Google API key for Gemini
            model: Gemini model to use (default: gemini-2.0-flash)
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            self.api_key = api_key
            self.model_name = model
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            raise AnalysisError(f"Failed to initialize Gemini: {str(e)}")

    def analyze_video(self, video_url: str, channel_name: str) -> Dict:
        """
        Analyze a YouTube video using Gemini

        Args:
            video_url: YouTube video URL
            channel_name: Name of the channel

        Returns:
            Dictionary with analysis results
        """
        prompt = self._create_analysis_prompt()

        try:
            print(f"  🔷 Using Gemini to analyze video...")
            response = self.model.generate_content([video_url, prompt])
            result = self._parse_response(response.text)

            # Add metadata
            result['video_url'] = video_url
            result['channel'] = channel_name
            result['analysis_method'] = f'gemini-{self.model_name}'

            # Ensure required fields exist
            if 'stocks' not in result:
                result['stocks'] = []
            if 'title' not in result:
                result['title'] = 'Unknown Title'
            if 'main_thesis' not in result:
                result['main_thesis'] = ''
            if 'published_at' not in result:
                result['published_at'] = ''

            return result

        except Exception as e:
            raise AnalysisError(f"Gemini analysis failed: {str(e)}")

    def _create_analysis_prompt(self) -> str:
        """Create the analysis prompt for Gemini"""
        return """
Analyze this investment video and extract ALL stock recommendations.

For EACH stock mentioned (spoken or shown visually on screen):
- ticker symbol (e.g., AAPL, TSLA, NVDA)
- company name
- action: "buy", "sell", or "hold"
- detailed reasoning (why this recommendation? what's the investment thesis?)
- confidence: a number between 0.0 to 1.0
- catalysts: list of specific reasons, events, or metrics mentioned
- price_target: if mentioned (optional)
- timeframe: investment timeframe if mentioned (optional)

Also extract:
- video title
- published date if visible
- main investment thesis or theme

Return ONLY valid JSON in this exact format (no markdown, no code blocks):
{
    "title": "video title",
    "published_at": "date if available",
    "main_thesis": "overall investment theme",
    "stocks": [
        {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "action": "buy",
            "reasoning": "detailed explanation",
            "confidence": 0.8,
            "catalysts": ["reason 1", "reason 2"],
            "price_target": "$200",
            "timeframe": "6-12 months"
        }
    ]
}

If no stocks are mentioned, return {"stocks": [], "title": "...", "main_thesis": "..."}.
"""

    def _parse_response(self, text: str) -> Dict:
        """
        Parse Gemini response and extract JSON

        Args:
            text: Raw response text

        Returns:
            Parsed dictionary
        """
        try:
            # Remove markdown code blocks if present
            text = re.sub(r'```json\s*|\s*```', '', text).strip()

            # Try to extract JSON if wrapped in other text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

            result = json.loads(text)
            return result

        except json.JSONDecodeError as e:
            raise AnalysisError(f"Failed to parse Gemini response as JSON: {str(e)}\nResponse: {text[:500]}")

    def is_available(self) -> bool:
        """Check if Gemini analyzer is available"""
        return bool(self.api_key)

    def get_cost_estimate(self, video_duration_seconds: int) -> float:
        """
        Estimate cost for analyzing a video

        Gemini 2.0 Flash pricing (approximate):
        - Input: ~$0.075 per 1M tokens
        - Output: ~$0.30 per 1M tokens
        - Video processing has additional costs

        Args:
            video_duration_seconds: Video length in seconds

        Returns:
            Estimated cost in USD
        """
        # Rough estimates based on video length
        if video_duration_seconds < 600:  # < 10 minutes
            return 0.01
        elif video_duration_seconds < 1800:  # < 30 minutes
            return 0.02
        elif video_duration_seconds < 3600:  # < 1 hour
            return 0.04
        else:  # > 1 hour
            return 0.06
