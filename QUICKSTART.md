# 🚀 Quick Start Guide

## Installation (3 minutes)

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
./setup.sh

# Activate environment
source venv/bin/activate

# Edit your API keys
nano .env  # or use your favorite editor

# Load environment variables
export $(cat .env | xargs)

# Run!
python main.py
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
export $(cat .env | xargs)
python main.py
```

## Getting API Keys (5 minutes)

### Gemini API (Recommended - Free Tier Available)
1. Go to https://ai.google.dev/
2. Click "Get API Key"
3. Create a new project or select existing
4. Copy your API key
5. Add to `.env`: `GEMINI_API_KEY=your_key_here`

### YouTube Data API (Required)
1. Go to https://console.cloud.google.com/
2. Create a new project or select existing
3. Enable "YouTube Data API v3"
4. Create credentials → API Key
5. Add to `.env`: `YOUTUBE_API_KEY=your_key_here`

### Optional: OpenAI or Anthropic
Only needed if not using Gemini or want transcript mode

**OpenAI**: https://platform.openai.com/api-keys
**Anthropic**: https://console.anthropic.com/

## First Run

```bash
# Quick test with default config
python main.py

# Custom output file
python main.py --output my_report.md

# Verbose mode (see everything)
python main.py --verbose
```

## What You'll See

```
📊 INVESTMENT ANALYSIS AGENT
============================================================
✅ Analyzer initialized: GeminiVideoAnalyzer
📊 Analysis mode: auto
📺 Channels to analyze: 3

📹 COLLECTING VIDEOS
------------------------------------------------------------
  Fetching from: InvestAnswers
    ✅ Found 5 videos
  ...

🔍 ANALYZING VIDEOS
------------------------------------------------------------
[1/15] Why I'm Buying Bitcoin NOW
  Channel: InvestAnswers
  🔷 Using Gemini to analyze video...
  ✅ Found 3 stock mentions
  💰 Cost: $0.0200

...

💰 Total estimated cost: $0.30

📊 AGGREGATING STOCKS
------------------------------------------------------------
  ✅ Found 12 unique stocks
  🔥 Multi-channel stocks: 4

📝 GENERATING REPORT
------------------------------------------------------------
  ✅ Report generated

✅ ANALYSIS COMPLETE
============================================================

📊 Videos analyzed: 15
🎯 Stocks found: 12
🔥 Multi-channel stocks: 4
💰 Total cost: $0.30

📄 Report saved to: /path/to/investment_report.md
```

## Customization

### Change Channels

Edit `config.yaml`:

```yaml
channels:
  - channel_id: "YOUR_CHANNEL_ID"
    name: "Channel Name"
```

### Change Analysis Mode

```yaml
analysis_mode: gemini  # or youtube_api, hybrid, auto
```

### Adjust Settings

```yaml
settings:
  videos_per_channel: 3  # Reduce for testing
  min_confidence: 0.7    # Higher = fewer, better results
  max_cost_per_run: 5.00 # Safety limit
```

## Troubleshooting

### "No API keys configured"
```bash
# Make sure .env exists and has keys
cat .env

# Load them into environment
export $(cat .env | xargs)

# Verify they're loaded
echo $GEMINI_API_KEY
```

### "Failed to fetch transcript"
- Switch to Gemini mode: `analysis_mode: gemini`
- Or use hybrid mode for automatic fallback

### "Quota exceeded"
- YouTube API has daily limits
- Reduce `videos_per_channel` in config.yaml

### Import errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## Next Steps

1. ✅ Run once with default settings
2. 📝 Review the generated report
3. ⚙️ Adjust `config.yaml` to your preferences
4. 🔄 Schedule regular runs (cron, etc.)
5. 📊 Build your own analysis on top of the output

## Tips

- Start with `videos_per_channel: 2` for testing
- Use `gemini-2.0-flash` for cost efficiency
- Set `min_confidence: 0.7` for quality over quantity
- Review the report format and customize in `src/agents/workflow.py`

## Need Help?

- Check the full README.md for detailed documentation
- Review example outputs in the examples/ folder (if available)
- Open an issue on GitHub

Happy investing! 📈
