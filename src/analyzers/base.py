"""Base classes and interfaces for video analyzers"""

from abc import ABC, abstractmethod
from typing import Dict, List
from enum import Enum


class AnalysisMode(Enum):
    """Supported analysis modes"""
    OLLAMA = "ollama"
    GEMINI = "gemini"
    YOUTUBE_API = "youtube_api"
    HYBRID = "hybrid"
    AUTO = "auto"


class AnalysisError(Exception):
    """Custom exception for analysis failures"""
    pass


class VideoAnalyzer(ABC):
    """Base class for all video analysis strategies"""

    @abstractmethod
    def analyze_video(self, video_url: str, channel_name: str) -> Dict:
        """
        Analyze a video and return structured stock data

        Args:
            video_url: YouTube video URL
            channel_name: Name of the channel

        Returns:
            Dictionary containing:
            {
                'video_url': str,
                'channel': str,
                'title': str,
                'published_at': str,
                'stocks': [
                    {
                        'ticker': str,
                        'company_name': str,
                        'action': str,  # 'buy', 'sell', or 'hold'
                        'reasoning': str,
                        'confidence': float,  # 0.0 to 1.0
                        'catalysts': List[str],
                        'price_target': str (optional),
                        'timeframe': str (optional)
                    }
                ],
                'main_thesis': str,
                'analysis_method': str
            }
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this analyzer is available (API keys configured, etc.)"""
        pass

    @abstractmethod
    def get_cost_estimate(self, video_duration_seconds: int) -> float:
        """Estimate cost in USD for analyzing a video"""
        pass

    def get_name(self) -> str:
        """Get the analyzer name"""
        return self.__class__.__name__
