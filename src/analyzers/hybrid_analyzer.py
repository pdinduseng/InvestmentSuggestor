"""Hybrid analyzer that combines multiple strategies"""

from typing import Dict
from .base import VideoAnalyzer, AnalysisError
from .gemini_analyzer import GeminiVideoAnalyzer
from .youtube_transcript_analyzer import YouTubeTranscriptAnalyzer


class HybridVideoAnalyzer(VideoAnalyzer):
    """Tries Gemini first, falls back to YouTube Transcript API if needed"""

    def __init__(self, gemini_analyzer: GeminiVideoAnalyzer, youtube_analyzer: YouTubeTranscriptAnalyzer):
        """
        Initialize hybrid analyzer

        Args:
            gemini_analyzer: Gemini video analyzer instance
            youtube_analyzer: YouTube transcript analyzer instance
        """
        self.gemini = gemini_analyzer
        self.youtube = youtube_analyzer

    def analyze_video(self, video_url: str, channel_name: str) -> Dict:
        """
        Analyze video using best available method

        Tries Gemini first for better context understanding,
        falls back to YouTube transcript API if Gemini fails.

        Args:
            video_url: YouTube video URL
            channel_name: Name of the channel

        Returns:
            Dictionary with analysis results
        """
        # Try Gemini first (better context, sees visuals)
        if self.gemini.is_available():
            try:
                print(f"  🔷 Trying Gemini analysis...")
                result = self.gemini.analyze_video(video_url, channel_name)
                result['analysis_method'] = 'hybrid-gemini'
                return result
            except AnalysisError as e:
                print(f"  ⚠️ Gemini failed, falling back to transcript: {str(e)}")
            except Exception as e:
                print(f"  ⚠️ Gemini error, falling back to transcript: {str(e)}")

        # Fallback to YouTube transcript
        if self.youtube.is_available():
            print(f"  📝 Using YouTube Transcript API...")
            result = self.youtube.analyze_video(video_url, channel_name)
            result['analysis_method'] = 'hybrid-transcript'
            return result

        raise AnalysisError("No available analysis method - both Gemini and YouTube transcript failed")

    def is_available(self) -> bool:
        """Check if at least one analyzer is available"""
        return self.gemini.is_available() or self.youtube.is_available()

    def get_cost_estimate(self, video_duration_seconds: int) -> float:
        """
        Get cost estimate for the preferred method

        Args:
            video_duration_seconds: Video length in seconds

        Returns:
            Estimated cost in USD
        """
        # Use Gemini cost since it's the preferred method
        if self.gemini.is_available():
            return self.gemini.get_cost_estimate(video_duration_seconds)
        return self.youtube.get_cost_estimate(video_duration_seconds)
