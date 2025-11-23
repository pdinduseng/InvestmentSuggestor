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

        # Get video metadata first (needed for cache naming)
        metadata = self._get_video_metadata(video_id)

        # Get transcript with improved caching
        print(f"  📝 Fetching transcript...")
        transcript_text = self._fetch_transcript(
            video_id,
            channel_name,
            metadata.get('title', 'Unknown'),
            metadata.get('published_at', '')
        )

        # Get historical context for this channel and stocks
        print(f"  📚 Loading historical context...")
        historical_context = self._load_historical_context(channel_name)

        # Analyze with LLM
        print(f"  🤖 Analyzing with {self.llm_provider}...")
        analysis = self._analyze_transcript_with_llm(transcript_text, historical_context)

        # Prepare result
        result = {
            'video_id': video_id,
            'video_url': video_url,
            'channel': channel_name,
            'title': metadata.get('title', 'Unknown Title'),
            'published_at': metadata.get('published_at', ''),
            'stocks': analysis.get('stocks', []),
            'main_thesis': analysis.get('main_thesis', ''),
            'analysis_method': f'youtube_transcript-{self.llm_provider}'
        }

        # Cache the analysis result
        self._save_analysis_cache(result)

        return result

    def _fetch_transcript(self, video_id: str, channel_name: str, video_title: str, published_at: str) -> str:
        """
        Fetch transcript from YouTube with improved caching

        Args:
            video_id: YouTube video ID
            channel_name: Name of the channel
            video_title: Title of the video
            published_at: Publication date

        Returns:
            Transcript text
        """
        import os
        import json
        from pathlib import Path
        from datetime import datetime
        import re

        # Create safe filename from title and channel
        def make_safe_filename(text: str, max_length: int = 50) -> str:
            """Convert text to safe filename"""
            # Remove special characters and limit length
            safe = re.sub(r'[^\w\s-]', '', text)
            safe = re.sub(r'[-\s]+', '_', safe)
            return safe[:max_length].strip('_')

        # Extract date from published_at (format: YYYY-MM-DD)
        date_str = 'unknown'
        if published_at:
            try:
                date_str = published_at.split('T')[0]  # Extract YYYY-MM-DD from ISO format
            except:
                date_str = 'unknown'

        # Create channel directory
        safe_channel = make_safe_filename(channel_name)
        cache_dir = Path('.cache/transcripts') / safe_channel

        # New cache file with descriptive name
        safe_title = make_safe_filename(video_title)
        cache_file = cache_dir / f"{date_str}_{safe_title}_{video_id}.json"

        # Also check old cache location for backward compatibility
        old_cache_file = Path('.cache/transcripts') / f"{video_id}.json"

        # Check new cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    print(f"  ✅ Using cached transcript: {cache_file.name}")
                    return cached['transcript']
            except Exception as e:
                print(f"  ⚠️ Cache read failed, fetching fresh: {e}")

        # Check old cache location
        if old_cache_file.exists():
            try:
                with open(old_cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    print(f"  ✅ Using cached transcript (legacy cache)")
                    transcript_text = cached['transcript']

                    # Migrate to new cache location
                    try:
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        with open(cache_file, 'w', encoding='utf-8') as f_new:
                            json.dump({
                                'video_id': video_id,
                                'channel': channel_name,
                                'title': video_title,
                                'published_at': published_at,
                                'transcript': transcript_text,
                                'fetched_at': cached.get('fetched_at', datetime.utcnow().isoformat()),
                                'language': cached.get('language', 'en')
                            }, f_new, indent=2)
                        print(f"  📦 Migrated to new cache format")
                    except Exception as e:
                        print(f"  ⚠️ Failed to migrate cache: {e}")

                    return transcript_text
            except Exception as e:
                print(f"  ⚠️ Legacy cache read failed: {e}")

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
                        'channel': channel_name,
                        'title': video_title,
                        'published_at': published_at,
                        'transcript': transcript_text,
                        'fetched_at': datetime.utcnow().isoformat(),
                        'language': 'en'
                    }, f, indent=2)
                print(f"  💾 Transcript cached: {cache_file.name}")
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

    def _analyze_transcript_with_llm(self, transcript: str, historical_context: str = "") -> Dict:
        """
        Analyze transcript using LLM with historical context

        Args:
            transcript: Video transcript text
            historical_context: Historical analysis data for this channel

        Returns:
            Analysis dictionary
        """
        historical_section = ""
        if historical_context:
            historical_section = f"""
HISTORICAL CONTEXT (Previous analyses from this channel):
{historical_context}

IMPORTANT - Use this historical context to:
1. Identify if stocks have been mentioned before and detect stance changes (e.g., HOLD → BUY)
2. Compare the PREVIOUS reasoning/thesis with the CURRENT one
3. If there's a change in action or thesis, explain WHAT CHANGED and WHY
4. Note if this is a new mention vs recurring recommendation
5. Identify if catalysts have changed or evolved
6. Provide more nuanced confidence based on consistency over time
"""

        prompt = f"""
Analyze this YouTube video transcript for stock investment recommendations.

{historical_section}

Extract ALL stocks mentioned with:
- ticker symbol (e.g., AAPL, TSLA)
- company name
- action: "buy", "sell", or "hold"
- detailed reasoning (investment thesis)
- confidence: 0.0 to 1.0
- catalysts: list of specific reasons
- price_target: if mentioned (optional)
- timeframe: if mentioned (optional)
- historical_note: if this stock was mentioned before, provide a brief summary (optional)
- thesis_evolution: if stance or reasoning changed, explain WHAT changed and WHY (optional)

IMPORTANT for historical stocks:
- If a stock appears in the historical context, compare the old vs new thesis
- Explain clearly what's different: new catalysts? changed fundamentals? different timeframe?
- Example: "Previously HOLD due to valuation concerns. Now BUY because earnings beat expectations and guidance raised, addressing previous concerns."

Also identify the main investment thesis or theme.

Return ONLY valid JSON in this format:
{{
    "main_thesis": "overall theme",
    "stocks": [
        {{
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "action": "buy",
            "reasoning": "detailed explanation of current thesis",
            "confidence": 0.8,
            "catalysts": ["reason 1", "reason 2"],
            "price_target": "$200",
            "timeframe": "6-12 months",
            "historical_note": "Previously recommended as HOLD (Nov 10)",
            "thesis_evolution": "Upgraded from HOLD to BUY. Previous concern was high valuation (P/E 35). Now justified by Q4 earnings beat (+15% YoY) and new AI chip announcement which wasn't factored in before. Price target raised from $175 to $200."
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

    def _save_analysis_cache(self, analysis: Dict) -> None:
        """
        Save analysis result to cache

        Args:
            analysis: Analysis result dictionary
        """
        import json
        from pathlib import Path
        from datetime import datetime
        import re

        try:
            # Create safe filename
            def make_safe_filename(text: str, max_length: int = 50) -> str:
                safe = re.sub(r'[^\w\s-]', '', text)
                safe = re.sub(r'[-\s]+', '_', safe)
                return safe[:max_length].strip('_')

            # Extract date from published_at
            date_str = 'unknown'
            if analysis.get('published_at'):
                try:
                    date_str = analysis['published_at'].split('T')[0]
                except:
                    date_str = 'unknown'

            # Create channel directory
            safe_channel = make_safe_filename(analysis['channel'])
            cache_dir = Path('.cache/reports') / safe_channel

            # Create filename
            safe_title = make_safe_filename(analysis['title'])
            video_id = analysis['video_id']
            cache_file = cache_dir / f"{date_str}_{safe_title}_{video_id}.json"

            # Save analysis
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                analysis_copy = analysis.copy()
                analysis_copy['cached_at'] = datetime.utcnow().isoformat()
                json.dump(analysis_copy, f, indent=2)

            print(f"  💾 Analysis cached: {cache_file.name}")

        except Exception as e:
            print(f"  ⚠️ Failed to cache analysis: {e}")

    def _load_historical_context(self, channel_name: str, days_back: int = 90) -> str:
        """
        Load historical analyses for a channel

        Args:
            channel_name: Name of the channel
            days_back: Number of days to look back

        Returns:
            Formatted historical context string
        """
        import json
        from pathlib import Path
        from datetime import datetime, timedelta
        import re

        try:
            # Create safe channel name
            def make_safe_filename(text: str, max_length: int = 50) -> str:
                safe = re.sub(r'[^\w\s-]', '', text)
                safe = re.sub(r'[-\s]+', '_', safe)
                return safe[:max_length].strip('_')

            safe_channel = make_safe_filename(channel_name)
            cache_dir = Path('.cache/reports') / safe_channel

            if not cache_dir.exists():
                return ""

            # Load all cached analyses
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            analyses = []

            for cache_file in cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # Check if within time window
                        if data.get('published_at'):
                            try:
                                pub_date = datetime.fromisoformat(data['published_at'].replace('Z', '+00:00'))
                                if pub_date < cutoff_date:
                                    continue
                            except:
                                pass

                        analyses.append(data)
                except Exception as e:
                    print(f"  ⚠️ Failed to load {cache_file.name}: {e}")
                    continue

            if not analyses:
                return ""

            # Sort by date (newest first)
            analyses.sort(key=lambda x: x.get('published_at', ''), reverse=True)

            # Build historical context summary
            context_lines = []
            context_lines.append(f"Found {len(analyses)} previous analyses from the last {days_back} days:\n")

            # Aggregate stock mentions
            stock_history = {}  # ticker -> list of mentions
            for analysis in analyses:
                date = analysis.get('published_at', 'unknown')[:10]
                title = analysis.get('title', 'Unknown')

                for stock in analysis.get('stocks', []):
                    ticker = stock.get('ticker', '').upper()
                    if not ticker:
                        continue

                    if ticker not in stock_history:
                        stock_history[ticker] = []

                    stock_history[ticker].append({
                        'date': date,
                        'title': title,
                        'action': stock.get('action', 'unknown'),
                        'confidence': stock.get('confidence', 0.0),
                        'reasoning': stock.get('reasoning', '')[:300],  # Increased for better context
                        'catalysts': stock.get('catalysts', []),
                        'price_target': stock.get('price_target', '')
                    })

            # Format stock history
            if stock_history:
                context_lines.append("\nSTOCK MENTION HISTORY:")
                for ticker, mentions in sorted(stock_history.items()):
                    context_lines.append(f"\n{ticker}: {len(mentions)} mention(s)")
                    for i, mention in enumerate(mentions[:3], 1):  # Show latest 3
                        context_lines.append(
                            f"  {i}. [{mention['date']}] Action: {mention['action'].upper()} "
                            f"| Confidence: {mention['confidence']:.0%}"
                        )
                        if mention['reasoning']:
                            context_lines.append(f"     Reasoning: {mention['reasoning']}")
                        if mention.get('catalysts'):
                            context_lines.append(f"     Catalysts: {', '.join(mention['catalysts'])}")
                        if mention.get('price_target'):
                            context_lines.append(f"     Price Target: {mention['price_target']}")

            return "\n".join(context_lines)

        except Exception as e:
            print(f"  ⚠️ Failed to load historical context: {e}")
            return ""

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
