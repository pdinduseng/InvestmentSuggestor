"""Ollama local model analyzer"""

import json
import re
from typing import Dict, Optional
from .base import VideoAnalyzer, AnalysisError


class OllamaAnalyzer(VideoAnalyzer):
    """Uses Ollama local models to analyze YouTube transcripts"""

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama analyzer

        Args:
            model: Ollama model to use (default: qwen2.5:7b)
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url

        try:
            import ollama
            self.client = ollama.Client(host=base_url)
        except ImportError:
            raise ImportError(
                "ollama package not installed. "
                "Install with: pip install ollama"
            )

        # Verify Ollama is running and model is available
        self._verify_setup()

    def _verify_setup(self):
        """Verify Ollama is running and model is available"""
        try:
            # List available models
            models = self.client.list()
            model_names = [m.model for m in models.models]

            if not any(self.model in name for name in model_names):
                print(f"⚠️ Model {self.model} not found locally.")
                print(f"   Available models: {', '.join(model_names)}")
                print(f"   The model will be pulled automatically on first use.")

        except Exception as e:
            raise AnalysisError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {str(e)}"
            )

    def analyze_video(self, video_url: str, channel_name: str) -> Dict:
        """
        Analyze a YouTube video using Ollama + transcript

        Args:
            video_url: YouTube video URL
            channel_name: Name of the channel

        Returns:
            Dictionary with analysis results
        """
        from ..utils import extract_video_id

        # Extract video ID
        try:
            video_id = extract_video_id(video_url)
        except ValueError as e:
            raise AnalysisError(f"Invalid YouTube URL: {str(e)}")

        # Get transcript
        print(f"  📝 Fetching transcript...")
        transcript_text = self._fetch_transcript(video_id)

        # Get video metadata
        metadata = self._get_video_metadata(video_id)

        # Analyze with Ollama
        print(f"  🤖 Analyzing with Ollama ({self.model})...")
        analysis = self._analyze_transcript_with_ollama(transcript_text)

        return {
            'video_url': video_url,
            'channel': channel_name,
            'title': metadata.get('title', 'Unknown Title'),
            'published_at': metadata.get('published_at', ''),
            'stocks': analysis.get('stocks', []),
            'main_thesis': analysis.get('main_thesis', ''),
            'analysis_method': f'ollama-{self.model}'
        }

    def _fetch_transcript(self, video_id: str) -> str:
        """Fetch transcript from YouTube (with caching)"""
        import os
        import json
        from pathlib import Path
        from datetime import datetime

        # Check cache first
        cache_dir = Path('.cache/transcripts')
        cache_file = cache_dir / f"{video_id}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    print(f"  ✅ Using cached transcript (fetched: {cached.get('fetched_at', 'unknown')})")
                    return cached['transcript']
            except Exception as e:
                print(f"  ⚠️ Cache read failed, fetching fresh: {e}")

        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            # Create API instance and fetch transcript
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=['en'])

            # Format transcript as plain text (snippets is a list of FetchedTranscriptSnippet objects)
            transcript_text = ' '.join([snippet.text for snippet in transcript.snippets])

            # Save to cache
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'video_id': video_id,
                        'transcript': transcript_text,
                        'fetched_at': datetime.utcnow().isoformat(),
                        'language': 'en'
                    }, f, indent=2)
                print(f"  💾 Transcript cached")
            except Exception as e:
                print(f"  ⚠️ Failed to cache transcript: {e}")

            return transcript_text

        except ImportError:
            raise ImportError("youtube-transcript-api not installed")
        except Exception as e:
            raise AnalysisError(f"Failed to fetch transcript: {str(e)}")

    def _get_video_metadata(self, video_id: str) -> Dict:
        """Get video metadata from YouTube API"""
        try:
            from googleapiclient.discovery import build
            import os

            api_key = os.getenv('YOUTUBE_API_KEY')
            if not api_key:
                return {'title': 'Unknown', 'published_at': '', 'duration': ''}

            youtube = build('youtube', 'v3', developerKey=api_key)
            request = youtube.videos().list(
                part='snippet,contentDetails',
                id=video_id
            )
            response = request.execute()

            if response['items']:
                snippet = response['items'][0]['snippet']
                content_details = response['items'][0]['contentDetails']

                return {
                    'title': snippet['title'],
                    'published_at': snippet['publishedAt'],
                    'duration': content_details.get('duration', '')
                }

        except Exception as e:
            print(f"  ⚠️ Could not fetch metadata: {str(e)}")

        return {'title': 'Unknown', 'published_at': '', 'duration': ''}

    def _analyze_transcript_with_ollama(self, transcript: str) -> Dict:
        """
        Analyze transcript using Ollama with structured output

        Args:
            transcript: Video transcript text

        Returns:
            Analysis dictionary
        """
        # Define JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "main_thesis": {
                    "type": "string",
                    "description": "Overall investment theme or main message"
                },
                "stocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "company_name": {"type": "string"},
                            "action": {
                                "type": "string",
                                "enum": ["buy", "sell", "hold"]
                            },
                            "reasoning": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0
                            },
                            "catalysts": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "price_target": {"type": "string"},
                            "timeframe": {"type": "string"}
                        },
                        "required": ["ticker", "company_name", "action", "reasoning", "confidence", "catalysts"]
                    }
                }
            },
            "required": ["main_thesis", "stocks"]
        }

        prompt = f"""Analyze this YouTube video transcript for stock investment recommendations.

Extract ALL stocks mentioned with:
- ticker symbol (e.g., AAPL, TSLA, NVDA)
- company name
- action: "buy", "sell", or "hold"
- detailed reasoning (investment thesis): Provide a COMPREHENSIVE explanation. Include financials, macro factors, and specific arguments made in the video. Do not be brief.
- confidence: 0.0 to 1.0 (how confident is the recommendation?)
- catalysts: list of specific reasons or events mentioned
- price_target: if mentioned (optional, use empty string if not mentioned)
- timeframe: investment timeframe if mentioned (optional, use empty string if not mentioned)

Also identify the main investment thesis or theme of the video.

Transcript:
{transcript[:8000]}

Return valid JSON only (no markdown, no code blocks).
"""

        try:
            # Use Ollama with structured output (format parameter)
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a financial analyst expert at extracting stock recommendations from investment content. Always respond with valid JSON.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                format=schema,  # Enforce JSON schema
                options={
                    'temperature': 0.1,  # Low temperature for consistent output
                    'num_predict': 4000  # Max tokens for response (increased for detailed reasoning)
                }
            )

            # Extract response
            response_text = response['message']['content']

            # Parse JSON
            result = self._parse_response(response_text)

            return result

        except Exception as e:
            raise AnalysisError(f"Ollama analysis failed: {str(e)}")

    def _parse_response(self, text: str) -> Dict:
        """Parse Ollama response and extract JSON"""
        try:
            # Remove markdown code blocks if present
            text = re.sub(r'```json\s*|\s*```', '', text).strip()

            # Try to extract JSON if wrapped in other text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

            result = json.loads(text)

            # Ensure required fields exist
            if 'stocks' not in result:
                result['stocks'] = []
            if 'main_thesis' not in result:
                result['main_thesis'] = ''

            return result

        except json.JSONDecodeError as e:
            raise AnalysisError(f"Failed to parse Ollama response as JSON: {str(e)}\nResponse: {text[:500]}")

    def is_available(self) -> bool:
        """Check if Ollama analyzer is available"""
        try:
            self.client.list()
            return True
        except:
            return False

    def get_cost_estimate(self, video_duration_seconds: int) -> float:
        """
        Estimate cost for analyzing a video

        Ollama is free (local), so cost is $0

        Args:
            video_duration_seconds: Video length in seconds

        Returns:
            Cost in USD (always 0 for Ollama)
        """
        return 0.0  # Free!

    def get_name(self) -> str:
        """Get analyzer name"""
        return f"OllamaAnalyzer({self.model})"
