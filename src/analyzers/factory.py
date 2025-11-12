"""Factory function for creating video analyzers"""

from typing import Dict
from .base import VideoAnalyzer, AnalysisMode
from .ollama_analyzer import OllamaAnalyzer
from .gemini_analyzer import GeminiVideoAnalyzer
from .youtube_transcript_analyzer import YouTubeTranscriptAnalyzer
from .hybrid_analyzer import HybridVideoAnalyzer


def create_analyzer(config: Dict) -> VideoAnalyzer:
    """
    Factory function to create the appropriate video analyzer

    Args:
        config: Configuration dictionary containing:
            - analysis_mode: 'ollama', 'gemini', 'youtube_api', 'hybrid', or 'auto'
            - ollama_model: Ollama model name (optional, default: qwen2.5:7b)
            - ollama_base_url: Ollama server URL (optional, default: http://localhost:11434)
            - gemini_api_key: Gemini API key (optional)
            - gemini_model: Gemini model name (optional)
            - llm_provider: 'openai', 'anthropic', or 'gemini' (for transcript mode)
            - openai_api_key: OpenAI API key (optional)
            - anthropic_api_key: Anthropic API key (optional)

    Returns:
        VideoAnalyzer instance

    Raises:
        ValueError: If configuration is invalid or no API keys provided
    """
    mode_str = config.get('analysis_mode', 'auto')
    mode = AnalysisMode(mode_str)

    if mode == AnalysisMode.OLLAMA:
        print("📊 Creating Ollama analyzer...")
        return OllamaAnalyzer(
            model=config.get('ollama_model', 'qwen2.5:7b'),
            base_url=config.get('ollama_base_url', 'http://localhost:11434')
        )

    elif mode == AnalysisMode.GEMINI:
        if not config.get('gemini_api_key'):
            raise ValueError("Gemini API key required for gemini mode")

        print("📊 Creating Gemini analyzer...")
        return GeminiVideoAnalyzer(
            api_key=config['gemini_api_key'],
            model=config.get('gemini_model', 'gemini-2.0-flash')
        )

    elif mode == AnalysisMode.YOUTUBE_API:
        llm_provider = config.get('llm_provider', 'openai')
        api_key = None

        if llm_provider == 'openai':
            api_key = config.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API key required for youtube_api mode with openai provider")
        elif llm_provider == 'anthropic':
            api_key = config.get('anthropic_api_key')
            if not api_key:
                raise ValueError("Anthropic API key required for youtube_api mode with anthropic provider")
        elif llm_provider == 'gemini':
            api_key = config.get('gemini_api_key')
            if not api_key:
                raise ValueError("Gemini API key required for youtube_api mode with gemini provider")
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")

        print(f"📊 Creating YouTube Transcript analyzer with {llm_provider}...")
        return YouTubeTranscriptAnalyzer(
            llm_provider=llm_provider,
            api_key=api_key,
            model=config.get('llm_model')
        )

    elif mode == AnalysisMode.HYBRID:
        print("📊 Creating Hybrid analyzer...")

        # Create Gemini analyzer
        gemini = None
        if config.get('gemini_api_key'):
            gemini = GeminiVideoAnalyzer(
                api_key=config['gemini_api_key'],
                model=config.get('gemini_model', 'gemini-2.0-flash')
            )
        else:
            print("  ⚠️ No Gemini API key provided")

        # Create YouTube transcript analyzer
        youtube = None
        llm_provider = config.get('llm_provider', 'openai')

        if llm_provider == 'openai' and config.get('openai_api_key'):
            youtube = YouTubeTranscriptAnalyzer(
                llm_provider='openai',
                api_key=config['openai_api_key']
            )
        elif llm_provider == 'anthropic' and config.get('anthropic_api_key'):
            youtube = YouTubeTranscriptAnalyzer(
                llm_provider='anthropic',
                api_key=config['anthropic_api_key']
            )
        elif llm_provider == 'gemini' and config.get('gemini_api_key'):
            youtube = YouTubeTranscriptAnalyzer(
                llm_provider='gemini',
                api_key=config['gemini_api_key']
            )
        else:
            print(f"  ⚠️ No {llm_provider} API key provided")

        if not gemini and not youtube:
            raise ValueError("Hybrid mode requires at least one analyzer to be configured")

        # Create fallback analyzer if one is missing
        if not gemini and youtube:
            return youtube
        if not youtube and gemini:
            return gemini

        return HybridVideoAnalyzer(gemini, youtube)

    elif mode == AnalysisMode.AUTO:
        # Auto-detect best available option
        # Priority: Ollama (free) > Gemini > OpenAI/Anthropic
        print("📊 Auto-detecting best analyzer...")

        # Try Ollama first (free!)
        try:
            ollama_analyzer = OllamaAnalyzer(
                model=config.get('ollama_model', 'qwen2.5:7b'),
                base_url=config.get('ollama_base_url', 'http://localhost:11434')
            )
            if ollama_analyzer.is_available():
                print("  ✅ Selected: Ollama (free local model)")
                return ollama_analyzer
        except Exception as e:
            print(f"  ⚠️ Ollama not available: {str(e)}")

        # Fallback to cloud providers
        if config.get('gemini_api_key'):
            print("  ✅ Selected: Gemini")
            return GeminiVideoAnalyzer(
                api_key=config['gemini_api_key'],
                model=config.get('gemini_model', 'gemini-2.0-flash')
            )

        llm_provider = config.get('llm_provider', 'openai')
        if llm_provider == 'openai' and config.get('openai_api_key'):
            print(f"  ✅ Selected: YouTube Transcript with OpenAI")
            return YouTubeTranscriptAnalyzer(
                llm_provider='openai',
                api_key=config['openai_api_key']
            )
        elif llm_provider == 'anthropic' and config.get('anthropic_api_key'):
            print(f"  ✅ Selected: YouTube Transcript with Anthropic")
            return YouTubeTranscriptAnalyzer(
                llm_provider='anthropic',
                api_key=config['anthropic_api_key']
            )
        else:
            raise ValueError(
                "No analyzers available. Please either:\n"
                "1. Install and run Ollama (free local option)\n"
                "2. Provide API keys for Gemini, OpenAI, or Anthropic"
            )

    else:
        raise ValueError(f"Unknown analysis mode: {mode_str}")
