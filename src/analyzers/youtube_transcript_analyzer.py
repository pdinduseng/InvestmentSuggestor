"""YouTube Transcript API based analyzer"""

import json
import re
from typing import Dict, Optional
from .base import VideoAnalyzer, AnalysisError


class YouTubeTranscriptAnalyzer(VideoAnalyzer):
    """Uses YouTube Transcript API + LLM for analysis"""

    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize YouTube Transcript analyzer

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

    def analyze_video(self, video_url: str, channel_name: str) -> Dict:
        """
        Analyze a YouTube video using transcript + LLM

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

        # Analyze with LLM
        print(f"  🤖 Analyzing with {self.llm_provider}...")
        analysis = self._analyze_transcript_with_llm(transcript_text)

        return {
            'video_url': video_url,
            'channel': channel_name,
            'title': metadata.get('title', 'Unknown Title'),
            'published_at': metadata.get('published_at', ''),
            'stocks': analysis.get('stocks', []),
            'main_thesis': analysis.get('main_thesis', ''),
            'analysis_method': f'youtube_transcript-{self.llm_provider}'
        }

    def _fetch_transcript(self, video_id: str) -> str:
        """
        Fetch transcript from YouTube

        Args:
            video_id: YouTube video ID

        Returns:
            Transcript text
        """
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
            raise ImportError(
                "youtube-transcript-api not installed. "
                "Install with: pip install youtube-transcript-api"
            )
        except Exception as e:
            raise AnalysisError(f"Failed to fetch transcript: {str(e)}")

    def _get_video_metadata(self, video_id: str) -> Dict:
        """
        Get video metadata from YouTube API

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary with title, published_at, duration
        """
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

    def _analyze_transcript_with_llm(self, transcript: str) -> Dict:
        """
        Analyze transcript using LLM

        Args:
            transcript: Video transcript text

        Returns:
            Analysis dictionary
        """
        prompt = f"""
Analyze this YouTube video transcript for stock investment recommendations.

Extract ALL stocks mentioned with:
- ticker symbol (e.g., AAPL, TSLA)
- company name
- action: "buy", "sell", or "hold"
- detailed reasoning (investment thesis)
- confidence: 0.0 to 1.0
- catalysts: list of specific reasons
- price_target: if mentioned (optional)
- timeframe: if mentioned (optional)

Also identify the main investment thesis or theme.

Return ONLY valid JSON in this format:
{{
    "main_thesis": "overall theme",
    "stocks": [
        {{
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "action": "buy",
            "reasoning": "detailed explanation",
            "confidence": 0.8,
            "catalysts": ["reason 1", "reason 2"],
            "price_target": "$200",
            "timeframe": "6-12 months"
        }}
    ]
}}

Transcript:
{transcript}
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
                    max_tokens=4096,
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
            raise AnalysisError(f"LLM analysis failed: {str(e)}")

    def is_available(self) -> bool:
        """Check if analyzer is available"""
        return bool(self.api_key and self.client)

    def get_cost_estimate(self, video_duration_seconds: int) -> float:
        """
        Estimate cost for analyzing a video

        Args:
            video_duration_seconds: Video length in seconds

        Returns:
            Estimated cost in USD
        """
        # Estimate tokens: ~2 tokens per second of speech
        estimated_tokens = video_duration_seconds * 2

        if self.llm_provider == "openai":
            # GPT-4o: $2.50/$10.00 per 1M tokens (input/output)
            input_cost = (estimated_tokens / 1_000_000) * 2.50
            output_cost = (500 / 1_000_000) * 10.00  # Estimate ~500 tokens output
            return input_cost + output_cost

        elif self.llm_provider == "anthropic":
            # Claude 3.5 Sonnet: $3/$15 per 1M tokens
            input_cost = (estimated_tokens / 1_000_000) * 3.00
            output_cost = (500 / 1_000_000) * 15.00
            return input_cost + output_cost

        elif self.llm_provider == "gemini":
            # Gemini 2.0 Flash: much cheaper
            input_cost = (estimated_tokens / 1_000_000) * 0.075
            output_cost = (500 / 1_000_000) * 0.30
            return input_cost + output_cost

        return 0.01  # Default estimate
