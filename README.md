# 📊 Investment Analysis Agent

An intelligent agent powered by **LangGraph** that analyzes YouTube videos from investment channels and generates comprehensive stock recommendation reports.

## ✨ Features

- **🎯 Multi-Channel Analysis**: Analyzes videos from multiple investment YouTube channels
- **🤖 Flexible AI Backends**: Supports Gemini, OpenAI, and Anthropic
- **📝 Dual Analysis Modes**:
  - **Gemini Native**: Analyzes videos directly (sees visuals + audio)
  - **YouTube Transcript**: Uses transcript API + LLM (verbatim text)
  - **Hybrid**: Best of both worlds with automatic fallback
- **🔥 Smart Aggregation**: Identifies stocks mentioned across multiple channels
- **📊 Confidence Scoring**: Filters recommendations by confidence level
- **💰 Cost Control**: Built-in cost estimation and limits
- **📄 Markdown Reports**: Generates detailed, formatted reports

## 🏗️ Architecture

The agent uses **LangGraph** for orchestrating a multi-step workflow:

```
Initialize → Collect Videos → Analyze Videos → Aggregate Stocks → Generate Report
```

Each node is modular and can be customized independently.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd InvestmentSuggestor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# At minimum, provide ONE of these:
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Required for fetching video lists:
YOUTUBE_API_KEY=your_youtube_key_here
```

### 3. Run

```bash
# Load environment variables
export $(cat .env | xargs)  # Linux/Mac
# On Windows: use `set` command or load manually

# Run the agent
python main.py

# Or with options:
python main.py --config config.yaml --output my_report.md --verbose
```

## 📋 Configuration

Edit `config.yaml` to customize the agent:

```yaml
# Analysis mode
analysis_mode: auto  # gemini, youtube_api, hybrid, or auto

# Channels to analyze
channels:
  - channel_id: "UCR6V6JF1y_xBOPYQFFvj83Q"
    name: "InvestAnswers"

# Settings
settings:
  # Video collection mode
  collection_mode: time_window  # or 'max_videos'
  lookback_hours: 48  # For time_window mode
  videos_per_channel: 5  # For max_videos mode

  min_confidence: 0.6
  max_cost_per_run: 10.00
```

### Collection Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `time_window` | Fetch videos from last N hours | Daily/scheduled analysis |
| `max_videos` | Fetch most recent N videos | Backfill or manual runs |

### Analysis Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `gemini` | Uses Gemini to analyze videos directly | Best accuracy, sees visuals |
| `youtube_api` | Uses transcript + LLM | Verbatim quotes, lower cost |
| `hybrid` | Tries Gemini, falls back to transcript | Maximum reliability |
| `auto` | Automatically selects best available | Recommended |

## 🎯 How It Works

### 1. Video Collection
- **Time Window Mode** (default): Fetches videos published in last N hours (e.g., 48 hours)
- **Max Videos Mode**: Fetches most recent N videos regardless of age
- Uses YouTube Data API with `publishedAfter` filter for time-based collection
- Includes video metadata (title, publish date, channel)

### 2. Video Analysis
- **Gemini Mode**: Sends YouTube URL directly to Gemini
  - Analyzes audio, video, and on-screen text
  - Identifies stocks from visual charts and tickers

- **Transcript Mode**: Fetches transcript via YouTube API
  - Gets verbatim text of what was said
  - Analyzes with OpenAI/Anthropic/Gemini

### 3. Stock Extraction
- Identifies ticker symbols (AAPL, TSLA, etc.)
- Extracts:
  - Investment action (buy/sell/hold)
  - Reasoning and thesis
  - Confidence score
  - Catalysts and price targets

### 4. Aggregation
- Combines mentions across channels
- Prioritizes stocks mentioned multiple times
- Detects sentiment alignment
- Ranks by priority score

### 5. Report Generation
- Creates markdown report with:
  - High-priority multi-channel stocks first
  - Per-channel analysis and reasoning
  - Single-channel mentions
  - Error log

## 📊 Example Output

```markdown
# 📊 Investment Analysis Report

## 🔥 HIGH PRIORITY (Multiple Channel Coverage)

### NVDA - NVIDIA Corporation
**Coverage:** 3 channels | **Sentiment:** ✅ Aligned (BUY) | **Avg Confidence:** 85%

**InvestAnswers** - [AI Chips: The Future...]
- **Action:** BUY
- **Reasoning:** Dominant position in AI infrastructure...
- **Catalysts:** Data center growth, AI adoption, pricing power

**Meet Kevin** - [Why I'm Buying NVDA...]
- **Action:** STRONG BUY
- **Reasoning:** Next-gen GPU architecture launching Q2...

---

## 📌 Single Channel Mentions

**AAPL** (Andrei Jikh) - BUY - Strong services growth...
```

## 🛠️ Development

### Project Structure

```
InvestmentSuggestor/
├── src/
│   ├── analyzers/          # Video analysis implementations
│   │   ├── base.py         # Abstract base class
│   │   ├── gemini_analyzer.py
│   │   ├── youtube_transcript_analyzer.py
│   │   ├── hybrid_analyzer.py
│   │   └── factory.py      # Analyzer factory
│   ├── agents/             # LangGraph workflow
│   │   └── workflow.py
│   └── utils/              # Utilities
│       ├── config.py
│       └── helpers.py
├── main.py                 # Entry point
├── config.yaml             # Configuration
└── requirements.txt
```

### Adding Custom Analyzers

Implement the `VideoAnalyzer` interface:

```python
from src.analyzers.base import VideoAnalyzer

class MyCustomAnalyzer(VideoAnalyzer):
    def analyze_video(self, video_url: str, channel_name: str) -> Dict:
        # Your implementation
        pass

    def is_available(self) -> bool:
        return True

    def get_cost_estimate(self, video_duration_seconds: int) -> float:
        return 0.0
```

## 💡 Tips

### Getting API Keys

- **Gemini**: https://ai.google.dev/ (Recommended - best for this use case)
- **OpenAI**: https://platform.openai.com/
- **Anthropic**: https://console.anthropic.com/
- **YouTube**: https://console.cloud.google.com/ → Enable YouTube Data API v3

### Finding Channel IDs

1. Go to the channel page
2. View page source (Ctrl+U)
3. Search for `"channelId"`
4. Or use: https://commentpicker.com/youtube-channel-id.php

### Cost Optimization

- Use `gemini-2.0-flash` (cheaper than Pro)
- Set `min_confidence: 0.7` to filter low-quality results
- Reduce `videos_per_channel` for testing
- Set `max_cost_per_run` to avoid surprises

## 🐛 Troubleshooting

### "No API keys configured"
- Make sure you've created `.env` file
- Export environment variables: `export $(cat .env | xargs)`
- Or set them directly: `export GEMINI_API_KEY=your_key`

### "Failed to fetch transcript"
- Some videos don't have transcripts
- Hybrid mode will fall back to Gemini automatically
- Or use `analysis_mode: gemini` to skip transcripts entirely

### "Quota exceeded" (YouTube API)
- YouTube API has daily quota limits (free tier: 10,000 units/day)
- Reduce `videos_per_channel` or spread requests over time
- Each video search costs ~100 units

## 📈 Future Enhancements

- [ ] Database storage for historical tracking
- [ ] Web UI dashboard
- [ ] Sentiment trend analysis over time
- [ ] Price correlation analysis
- [ ] Automated scheduling (cron jobs)
- [ ] Email/Slack notifications
- [ ] Multi-language support
- [ ] Custom scoring algorithms

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional LLM providers
- Better stock symbol extraction
- Enhanced report formatting
- Performance optimizations
- Test coverage

## 🙏 Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- Google Gemini, OpenAI, Anthropic - LLM providers

---

**⚠️ Disclaimer**: This tool is for informational purposes only. Not financial advice. Always do your own research before making investment decisions.
