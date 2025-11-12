# 🏗️ Architecture Deep Dive

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Investment Analysis Agent                 │
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   Config    │───▶│   Factory    │───▶│   Analyzer    │  │
│  │ (YAML+Env)  │    │              │    │  (Strategy)   │  │
│  └─────────────┘    └──────────────┘    └───────┬───────┘  │
│                                                   │          │
│                    ┌──────────────────────────────┘          │
│                    │                                         │
│          ┌─────────▼────────┐                                │
│          │  LangGraph Agent  │                                │
│          └──────────┬────────┘                                │
│                     │                                         │
└─────────────────────┼─────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
   ┌────▼────┐               ┌──────▼──────┐
   │ YouTube │               │ AI Providers │
   │   API   │               │ (LLM APIs)  │
   └─────────┘               └─────────────┘
```

## Component Breakdown

### 1. Configuration Layer

```
┌──────────────────────────────────────────┐
│          Configuration System             │
├──────────────────────────────────────────┤
│                                           │
│  config.yaml                              │
│    ├─ analysis_mode: auto                │
│    ├─ api_keys: ${ENV_VARS}              │
│    ├─ channels: [...]                    │
│    └─ settings: {...}                    │
│                                           │
│  Environment Variables (.env)             │
│    ├─ GEMINI_API_KEY                     │
│    ├─ YOUTUBE_API_KEY                    │
│    └─ ...                                │
│                                           │
│  utils/config.py                          │
│    ├─ load_config()                      │
│    ├─ _substitute_env_vars()            │
│    └─ _validate_config()                │
└──────────────────────────────────────────┘
```

### 2. Analyzer Layer (Strategy Pattern)

```
                    ┌─────────────────┐
                    │ VideoAnalyzer   │
                    │   (Abstract)    │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
    │   Gemini    │  │  YouTube    │  │   Hybrid    │
    │  Analyzer   │  │ Transcript  │  │  Analyzer   │
    │             │  │  Analyzer   │  │             │
    └─────────────┘  └─────────────┘  └─────────────┘
         │                 │                 │
         │                 │                 │
    ┌────▼────┐       ┌────▼────┐      ┌────▼────┐
    │ Gemini  │       │YouTube  │      │  Both   │
    │   API   │       │API + LLM│      │ (Smart) │
    └─────────┘       └─────────┘      └─────────┘

Factory Pattern:
  create_analyzer(config) → VideoAnalyzer
```

### 3. LangGraph Workflow

```
                    ┌─────────────────────────────┐
                    │      Agent State            │
                    │  (Shared across all nodes)  │
                    └──────────┬──────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼────┐           ┌─────▼─────┐         ┌─────▼─────┐
   │ config  │           │ analyzer  │         │ channels  │
   └─────────┘           └───────────┘         └───────────┘
        │                      │                      │
   ┌────▼────┐           ┌─────▼─────┐         ┌─────▼─────┐
   │video_   │           │video_     │         │aggregated_│
   │urls     │           │analyses   │         │stocks     │
   └─────────┘           └───────────┘         └───────────┘

Workflow Graph:

    START
      │
      ▼
  ┌─────────────┐
  │ Initialize  │  • Load config
  │             │  • Create analyzer
  └──────┬──────┘  • Validate setup
         │
         ▼
  ┌─────────────┐
  │  Collect    │  • Query YouTube API
  │  Videos     │  • Get recent uploads
  └──────┬──────┘  • Extract metadata
         │
         ▼
  ┌─────────────┐
  │  Analyze    │  • Process each video
  │  Videos     │  • Extract stocks
  └──────┬──────┘  • Track costs
         │
         ▼
  ┌─────────────┐
  │ Aggregate   │  • Group by ticker
  │  Stocks     │  • Calculate scores
  └──────┬──────┘  • Detect patterns
         │
         ▼
  ┌─────────────┐
  │  Generate   │  • Create markdown
  │  Report     │  • Format output
  └──────┬──────┘  • Save to file
         │
         ▼
       END
```

### 4. Data Models

```python
# Stock Mention
{
    'ticker': str,           # e.g., 'AAPL'
    'company_name': str,     # e.g., 'Apple Inc.'
    'action': str,           # 'buy', 'sell', 'hold'
    'reasoning': str,        # Investment thesis
    'confidence': float,     # 0.0 to 1.0
    'catalysts': [str],      # List of reasons
    'price_target': str?,    # Optional
    'timeframe': str?        # Optional
}

# Video Analysis
{
    'video_url': str,
    'channel': str,
    'title': str,
    'published_at': str,
    'stocks': [StockMention],
    'main_thesis': str,
    'analysis_method': str
}

# Aggregated Stock
{
    'ticker': str,
    'company_name': str,
    'num_channels': int,
    'sentiment_aligned': bool,
    'dominant_action': str,
    'avg_confidence': float,
    'priority_score': float,
    'mentions': [
        {
            'channel': str,
            'video_url': str,
            'video_title': str,
            'action': str,
            'reasoning': str,
            'confidence': float,
            'catalysts': [str]
        }
    ]
}
```

### 5. Priority Scoring Algorithm

```
For each stock:

priority_score = 
    (num_channels × 3) +           # Multi-channel bonus
    (sentiment_aligned × 2) +      # All same action?
    avg_confidence                 # Average confidence

Example:
  NVDA mentioned in 3 channels (all BUY, avg conf 0.85):
  score = (3 × 3) + (1 × 2) + 0.85 = 11.85

  TSLA mentioned in 2 channels (mixed, avg conf 0.70):
  score = (2 × 3) + (0 × 2) + 0.70 = 6.70

Sort stocks by priority_score (descending)
```

## Sequence Diagrams

### Gemini Mode

```
User ─────► main.py ─────► load_config()
                │
                ├─────► create_analyzer(config)
                │           │
                │           └─────► GeminiVideoAnalyzer
                │
                ├─────► LangGraph.invoke()
                │           │
                │           ├─► collect_videos()
                │           │       └─► YouTube API
                │           │
                │           ├─► analyze_videos()
                │           │       │
                │           │       └─► For each video:
                │           │               │
                │           │               └─► Gemini API
                │           │                   (video URL + prompt)
                │           │                       │
                │           │                       └─► Structured JSON
                │           │
                │           ├─► aggregate_stocks()
                │           │       └─► Calculate priorities
                │           │
                │           └─► generate_report()
                │                   └─► Markdown file
                │
                └─────► Display results
```

### YouTube Transcript Mode

```
User ─────► main.py ─────► YouTubeTranscriptAnalyzer
                │
                └─────► For each video:
                            │
                            ├─► YouTube Transcript API
                            │       └─► Get text transcript
                            │
                            ├─► LLM (OpenAI/Anthropic/Gemini)
                            │       └─► Analyze transcript text
                            │               │
                            │               └─► Extract stocks
                            │
                            └─► Return structured data
```

### Hybrid Mode (Fallback)

```
Hybrid Analyzer
    │
    ├─► Try: Gemini API
    │       ├─► Success? ✅ Return result
    │       └─► Failed? ⚠️  Continue
    │
    └─► Fallback: YouTube Transcript
            └─► Success? ✅ Return result
```

## Error Handling Flow

```
┌──────────────────────────────────────────────┐
│              Error Handling                  │
├──────────────────────────────────────────────┤
│                                               │
│  Level 1: Analyzer Level                     │
│    • API errors caught                       │
│    • Retry logic (if applicable)            │
│    • Raise AnalysisError                     │
│                                               │
│  Level 2: Workflow Node Level                │
│    • Catch AnalysisError                     │
│    • Append to state['errors']              │
│    • Continue with next video                │
│                                               │
│  Level 3: Application Level                  │
│    • Catch fatal errors                      │
│    • Log traceback (if --verbose)           │
│    • Exit gracefully                         │
│                                               │
│  Reporting:                                   │
│    • All errors collected in state           │
│    • Included in final report                │
│    • User sees what failed and why           │
└──────────────────────────────────────────────┘
```

## Configuration Flow

```
Load config.yaml
    │
    ├─► Parse YAML
    │
    ├─► Substitute ${ENV_VARS}
    │       └─► Read from environment
    │
    ├─► Validate structure
    │       ├─► Check required fields
    │       ├─► Validate modes
    │       └─► Validate channels
    │
    ├─► Check API keys
    │       └─► Warn if missing
    │
    └─► Return config dict
```

## Extension Points

### Adding New Analyzer

```python
# 1. Create new analyzer
class MyCustomAnalyzer(VideoAnalyzer):
    def analyze_video(self, url, channel):
        # Your implementation
        pass

# 2. Register in factory
def create_analyzer(config):
    if config['analysis_mode'] == 'custom':
        return MyCustomAnalyzer(...)

# 3. Update config.yaml
analysis_mode: custom
```

### Adding New Workflow Node

```python
# 1. Define node function
def my_new_node(state: AgentState) -> Dict:
    # Your logic
    return {"new_field": value}

# 2. Add to graph
workflow.add_node("my_node", my_new_node)

# 3. Connect edges
workflow.add_edge("previous_node", "my_node")
workflow.add_edge("my_node", "next_node")
```

### Adding New Data Source

```python
# Example: Add Twitter analysis
class TwitterAnalyzer(VideoAnalyzer):
    def analyze_video(self, tweet_url, author):
        # Fetch tweet
        # Analyze with LLM
        # Return same format
        pass
```

## Performance Considerations

```
Bottlenecks:
1. API Rate Limits
   • YouTube API: 10,000 units/day (free)
   • LLM APIs: varies by provider

2. API Latency
   • Gemini: ~2-5s per video
   • Transcript + LLM: ~3-8s per video

3. Sequential Processing
   • Videos analyzed one at a time
   • Future: parallel processing

Optimizations:
1. Cost limits prevent runaway costs
2. Confidence filtering reduces noise
3. Caching could be added for repeated runs
4. Parallel video analysis (ThreadPoolExecutor)
```

## Security Considerations

```
✅ Implemented:
• Environment variables for secrets
• .env excluded from git
• No hardcoded credentials
• Input validation

⚠️ Consider:
• API key rotation
• Rate limiting
• Request signing
• Audit logging
```

---

**This architecture is designed to be:**
- **Modular**: Easy to swap components
- **Extensible**: Add new features without breaking existing code
- **Testable**: Mock interfaces for unit tests
- **Observable**: Track state through workflow
- **Maintainable**: Clear separation of concerns
