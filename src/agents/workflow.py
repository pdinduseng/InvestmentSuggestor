"""LangGraph workflow for investment analysis"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from ..analyzers import VideoAnalyzer, create_analyzer
from ..utils import load_config


class AgentState(TypedDict):
    """State shared across all nodes in the workflow"""
    config: Dict
    analyzer: VideoAnalyzer
    channels: List[Dict[str, str]]
    video_urls: List[Dict[str, str]]
    video_analyses: List[Dict]
    aggregated_stocks: List[Dict]
    final_report: str
    errors: List[str]
    total_cost: float


def initialize_agent(state: AgentState) -> Dict:
    """
    Initialize the agent and create video analyzer

    Args:
        state: Current agent state

    Returns:
        Updated state dict
    """
    print("=" * 60)
    print("🚀 INVESTMENT ANALYSIS AGENT")
    print("=" * 60)

    config = state['config']

    # Create the appropriate analyzer
    try:
        analyzer = create_analyzer(config)
    except Exception as e:
        raise RuntimeError(f"Failed to create analyzer: {str(e)}")

    if not analyzer.is_available():
        raise RuntimeError("Video analyzer is not available. Please check your API keys.")

    print(f"✅ Analyzer initialized: {analyzer.get_name()}")
    print(f"📊 Analysis mode: {config.get('analysis_mode', 'auto')}")
    print(f"📺 Channels to analyze: {len(config['channels'])}")
    print()

    return {
        "analyzer": analyzer,
        "channels": config['channels'],
        "total_cost": 0.0,
        "errors": []
    }


def collect_videos(state: AgentState) -> Dict:
    """
    Fetch recent video URLs from YouTube channels

    Supports two modes:
    - time_window: Fetch videos from last N hours
    - max_videos: Fetch most recent N videos

    Args:
        state: Current agent state

    Returns:
        Updated state with video URLs
    """
    print("📹 COLLECTING VIDEOS")
    print("-" * 60)

    try:
        from googleapiclient.discovery import build
        from datetime import datetime, timedelta
        import os
    except ImportError:
        raise ImportError(
            "google-api-python-client not installed. "
            "Install with: pip install google-api-python-client"
        )

    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    if not youtube_api_key:
        print("⚠️ YOUTUBE_API_KEY not set, using placeholder data")
        return {"video_urls": []}

    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    video_urls = []
    settings = state['config'].get('settings', {})

    # Get collection mode settings
    collection_mode = settings.get('collection_mode', 'time_window')
    default_lookback_hours = settings.get('lookback_hours', 48)
    max_videos = settings.get('videos_per_channel', 5)
    absolute_max = settings.get('max_videos_per_channel', 50)

    if collection_mode == 'time_window':
        print(f"  Mode: Time window (default: {default_lookback_hours} hours, per-channel overrides may apply)")
    else:
        print(f"  Mode: Max videos ({max_videos} most recent per channel)")

    print()

    for channel in state['channels']:
        try:
            # Check for channel-specific lookback hours
            channel_lookback = channel.get('lookback_hours', default_lookback_hours)

            print(f"  Fetching from: {channel['name']}", end='')

            # Calculate time window for this channel
            published_after = None
            if collection_mode == 'time_window':
                cutoff_time = datetime.utcnow() - timedelta(hours=channel_lookback)
                published_after = cutoff_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                print(f" (last {channel_lookback} hours)")
            else:
                print()

            # Build request based on mode
            request_params = {
                'channelId': channel['channel_id'],
                'part': 'id,snippet',
                'order': 'date',
                'maxResults': absolute_max,  # Use absolute max as safety limit
                'type': 'video',
                'videoDefinition': 'any'
            }

            # Add time filter if in time_window mode
            if collection_mode == 'time_window' and published_after:
                request_params['publishedAfter'] = published_after

            request = youtube.search().list(**request_params)
            response = request.execute()

            # In max_videos mode, limit the results
            items = response['items']
            if collection_mode == 'max_videos':
                items = items[:max_videos]

            for item in items:
                video_id = item['id']['videoId']
                video_urls.append({
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'channel': channel['name'],
                    'title': item['snippet']['title'],
                    'published_at': item['snippet'].get('publishedAt', '')
                })

            print(f"    ✅ Found {len(items)} videos")

        except Exception as e:
            error_msg = f"Error fetching videos from {channel['name']}: {str(e)}"
            state['errors'].append(error_msg)
            print(f"    ❌ {error_msg}")

    print(f"\n📊 Total videos found: {len(video_urls)}")
    print()

    return {"video_urls": video_urls}


def analyze_videos(state: AgentState) -> Dict:
    """
    Analyze videos using the configured analyzer

    Args:
        state: Current agent state

    Returns:
        Updated state with video analyses
    """
    print("🔍 ANALYZING VIDEOS")
    print("-" * 60)

    analyzer = state['analyzer']
    analyses = []
    total_cost = state.get('total_cost', 0.0)

    settings = state['config'].get('settings', {})
    max_cost = settings.get('max_cost_per_run', float('inf'))
    min_confidence = settings.get('min_confidence', 0.0)

    for i, video_info in enumerate(state['video_urls'], 1):
        print(f"\n[{i}/{len(state['video_urls'])}] {video_info['title']}")
        print(f"  Channel: {video_info['channel']}")

        # Check cost limit
        estimated_cost = analyzer.get_cost_estimate(600)  # Assume 10min videos
        if total_cost + estimated_cost > max_cost:
            print(f"\n💰 Cost limit reached (${max_cost:.2f}), stopping analysis")
            break

        try:
            analysis = analyzer.analyze_video(
                video_url=video_info['url'],
                channel_name=video_info['channel']
            )

            # Filter stocks by confidence
            if 'stocks' in analysis:
                original_count = len(analysis['stocks'])
                analysis['stocks'] = [
                    stock for stock in analysis['stocks']
                    if stock.get('confidence', 0.0) >= min_confidence
                ]
                filtered_count = len(analysis['stocks'])

                if filtered_count < original_count:
                    print(f"  📊 Filtered stocks: {original_count} → {filtered_count} (min confidence: {min_confidence})")

            analyses.append(analysis)
            total_cost += estimated_cost

            print(f"  ✅ Found {len(analysis.get('stocks', []))} stock mentions")
            print(f"  💰 Cost: ${estimated_cost:.4f}")

        except Exception as e:
            error_msg = f"Analysis failed for {video_info['title']}: {str(e)}"
            state['errors'].append(error_msg)
            print(f"  ❌ {error_msg}")

    print(f"\n💰 Total estimated cost: ${total_cost:.2f}")
    print()

    return {
        "video_analyses": analyses,
        "total_cost": total_cost
    }


def aggregate_stocks(state: AgentState) -> Dict:
    """
    Aggregate stock mentions across all channels

    Args:
        state: Current agent state

    Returns:
        Updated state with aggregated stocks
    """
    print("📊 AGGREGATING STOCKS")
    print("-" * 60)

    stock_map = {}  # ticker -> aggregated data

    for analysis in state['video_analyses']:
        channel = analysis.get('channel', 'Unknown')
        video_url = analysis.get('video_url', '')
        video_title = analysis.get('title', 'Unknown')

        for stock in analysis.get('stocks', []):
            ticker = stock.get('ticker', '').upper()
            if not ticker:
                continue

            if ticker not in stock_map:
                stock_map[ticker] = {
                    'ticker': ticker,
                    'company_name': stock.get('company_name', ticker),
                    'mentions': [],
                    'channels': set()
                }

            stock_map[ticker]['mentions'].append({
                'channel': channel,
                'video_url': video_url,
                'video_title': video_title,
                'action': stock.get('action', 'unknown'),
                'reasoning': stock.get('reasoning', ''),
                'confidence': stock.get('confidence', 0.0),
                'catalysts': stock.get('catalysts', []),
                'price_target': stock.get('price_target'),
                'timeframe': stock.get('timeframe')
            })

            stock_map[ticker]['channels'].add(channel)

    # Calculate priority scores
    for ticker, data in stock_map.items():
        mentions = data['mentions']
        num_channels = len(data['channels'])

        # Check sentiment alignment (all same action?)
        actions = [m['action'] for m in mentions]
        aligned = len(set(actions)) == 1
        dominant_action = max(set(actions), key=actions.count)

        # Average confidence
        avg_confidence = sum(m['confidence'] for m in mentions) / len(mentions)

        # Priority score
        priority = (
            num_channels * 3 +  # Multi-channel bonus
            (2 if aligned else 0) +  # Alignment bonus
            avg_confidence  # Confidence score
        )

        data['num_channels'] = num_channels
        data['sentiment_aligned'] = aligned
        data['dominant_action'] = dominant_action
        data['avg_confidence'] = avg_confidence
        data['priority_score'] = priority
        data['channels'] = list(data['channels'])  # Convert set to list

    # Sort by priority
    sorted_stocks = sorted(
        stock_map.values(),
        key=lambda x: x['priority_score'],
        reverse=True
    )

    print(f"  ✅ Found {len(sorted_stocks)} unique stocks")
    print(f"  🔥 Multi-channel stocks: {sum(1 for s in sorted_stocks if s['num_channels'] > 1)}")
    print()

    return {"aggregated_stocks": sorted_stocks}


def generate_report(state: AgentState) -> Dict:
    """
    Generate final markdown report

    Args:
        state: Current agent state

    Returns:
        Updated state with final report
    """
    print("📝 GENERATING REPORT")
    print("-" * 60)

    report = []

    # Header
    report.append("# 📊 Investment Analysis Report\n")
    report.append(f"*Analyzed {len(state['video_analyses'])} videos across {len(state['channels'])} channels*\n")
    report.append(f"*Total cost: ${state['total_cost']:.2f}*\n")
    report.append("---\n")

    # High priority stocks (multiple channels)
    multi_channel = [s for s in state['aggregated_stocks'] if s['num_channels'] > 1]

    if multi_channel:
        report.append("## 🔥 HIGH PRIORITY (Multiple Channel Coverage)\n")

        for stock in multi_channel:
            # Stock header
            report.append(f"### {stock['ticker']} - {stock['company_name']}\n")
            report.append(f"**Coverage:** {stock['num_channels']} channels | ")
            report.append(f"**Sentiment:** {'✅ Aligned' if stock['sentiment_aligned'] else '⚠️ Mixed'} ({stock['dominant_action'].upper()}) | ")
            report.append(f"**Avg Confidence:** {stock['avg_confidence']:.0%}\n")

            # Per-channel details
            for mention in stock['mentions']:
                report.append(f"\n**{mention['channel']}** - [{mention['video_title'][:60]}...]({mention['video_url']})")
                report.append(f"- **Action:** {mention['action'].upper()}")
                report.append(f"- **Reasoning:** {mention['reasoning']}")
                report.append(f"- **Confidence:** {mention['confidence']:.0%}")

                if mention['catalysts']:
                    report.append(f"- **Catalysts:** {', '.join(mention['catalysts'])}")

                if mention['price_target']:
                    report.append(f"- **Price Target:** {mention['price_target']}")

                if mention['timeframe']:
                    report.append(f"- **Timeframe:** {mention['timeframe']}")

                report.append("")

            report.append("---\n")

    # Single channel mentions
    single_channel = [s for s in state['aggregated_stocks'] if s['num_channels'] == 1]

    if single_channel:
        report.append("## 📌 Single Channel Mentions\n")

        for stock in single_channel[:20]:  # Limit to top 20
            mention = stock['mentions'][0]
            report.append(
                f"**{stock['ticker']}** ({mention['channel']}) - "
                f"{mention['action'].upper()} - "
                f"{mention['reasoning']}..."
            )
            if mention['video_url']:
                report.append(f" [Watch]({mention['video_url']})")
            report.append("\n")

        if len(single_channel) > 20:
            report.append(f"\n*... and {len(single_channel) - 20} more*\n")

    # Errors section
    if state['errors']:
        report.append("\n---\n")
        report.append("## ⚠️ Errors Encountered\n")
        for error in state['errors']:
            report.append(f"- {error}\n")

    final_report = "\n".join(report)

    print("  ✅ Report generated")
    print()

    return {"final_report": final_report}


def create_investment_agent():
    """
    Create the LangGraph workflow for investment analysis

    Returns:
        Compiled LangGraph application
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("initialize", initialize_agent)
    workflow.add_node("collect_videos", collect_videos)
    workflow.add_node("analyze_videos", analyze_videos)
    workflow.add_node("aggregate_stocks", aggregate_stocks)
    workflow.add_node("generate_report", generate_report)

    # Define edges (sequential flow)
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "collect_videos")
    workflow.add_edge("collect_videos", "analyze_videos")
    workflow.add_edge("analyze_videos", "aggregate_stocks")
    workflow.add_edge("aggregate_stocks", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()
