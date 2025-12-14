"""LangGraph workflow for investment analysis"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from ..analyzers import VideoAnalyzer, create_analyzer
from ..utils import load_config


class AgentState(TypedDict):
    """State shared across all nodes in the workflow"""
    config: Dict
    # analyzer: VideoAnalyzer (removed to avoid serialization issues)
    channels: List[Dict[str, str]]
    video_urls: List[Dict[str, str]]
    video_analyses: List[Dict]
    failed_json_parses: List[Dict]  # Videos with JSON parsing errors
    aggregated_stocks: List[Dict]
    market_data: Dict  # ticker -> market data mapping
    critic_result: Dict
    investigation_results: List[Dict]
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
        Updated state with video analyses and any JSON parsing failures
    """
    print("🔍 ANALYZING VIDEOS")
    print("-" * 60)

    # Create analyzer (re-instantiate to avoid storing in state)
    try:
        analyzer = create_analyzer(state['config'])
    except Exception as e:
        error_msg = f"Failed to recreate analyzer: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "video_analyses": [],
            "failed_json_parses": [],
            "total_cost": 0.0,
            "errors": state.get('errors', []) + [error_msg]
        }

    analyses = []
    failed_json = []
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
            error_str = str(e)

            # Check if this is a JSON parsing error
            if "Failed to parse" in error_str and "JSON" in error_str:
                # Extract the malformed JSON from error message if possible
                raw_response = None
                if "Response: " in error_str:
                    try:
                        raw_response = error_str.split("Response: ", 1)[1]
                    except:
                        raw_response = None

                print(f"  ⚠️  JSON parsing error - will attempt repair")
                failed_json.append({
                    'video_info': video_info,
                    'error': error_str,
                    'raw_response': raw_response,
                    'analyzer_type': analyzer.get_name()
                })
                total_cost += estimated_cost  # Count the cost even if parsing failed
            else:
                # Regular error, not JSON related
                error_msg = f"Analysis failed for {video_info['title']}: {error_str}"
                state['errors'].append(error_msg)
                print(f"  ❌ {error_msg}")

    print(f"\n💰 Total estimated cost: ${total_cost:.2f}")

    if failed_json:
        print(f"⚠️  {len(failed_json)} video(s) with JSON parsing errors will be repaired")

    print()

    return {
        "video_analyses": analyses,
        "failed_json_parses": failed_json,
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
                'timeframe': stock.get('timeframe'),
                'historical_note': stock.get('historical_note', ''),
                'thesis_evolution': stock.get('thesis_evolution', '')
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


def fetch_market_data(state: AgentState) -> Dict:
    """
    Fetch real-time market data from Yahoo Finance for all stocks

    Args:
        state: Current agent state

    Returns:
        Updated state with market data
    """
    print("💹 FETCHING MARKET DATA")
    print("-" * 60)

    aggregated_stocks = state['aggregated_stocks']

    if not aggregated_stocks:
        print("  ℹ️  No stocks to fetch data for")
        print()
        return {"market_data": {}}

    # Check if yfinance is available
    try:
        import yfinance as yf
    except ImportError:
        print("  ⚠️  yfinance not installed. Skipping market data fetch.")
        print("  Install with: pip install yfinance")
        print()
        return {"market_data": {}}

    market_data_map = {}
    tickers = [stock['ticker'] for stock in aggregated_stocks]

    print(f"  Fetching data for {len(tickers)} stocks...")

    for ticker in tickers:
        try:
            print(f"    {ticker}...", end=' ')
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="3mo")

            # Get recent news
            try:
                news = stock.news[:3]  # Top 3 news items
            except:
                news = []

            # Calculate key metrics
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price and not history.empty:
                current_price = history['Close'].iloc[-1]

            # Calculate recent performance
            if not history.empty and len(history) > 20:
                month_ago_price = history['Close'].iloc[-21]  # ~1 month
                month_performance = ((current_price - month_ago_price) / month_ago_price * 100) if month_ago_price else None
            else:
                month_performance = None

            market_data = {
                'ticker': ticker,
                'current_price': current_price,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'analyst_target_price': info.get('targetMeanPrice'),
                'analyst_target_high': info.get('targetHighPrice'),
                'analyst_target_low': info.get('targetLowPrice'),
                'recommendation_mean': info.get('recommendationMean'),  # 1=Strong Buy, 5=Strong Sell
                'month_performance': month_performance,
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'news_headlines': [
                    {
                        'title': item.get('title', ''),
                        'publisher': item.get('publisher', ''),
                    }
                    for item in news[:3]
                ] if news else []
            }

            market_data_map[ticker] = market_data
            print("✅")

        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            state['errors'].append(f"Failed to fetch market data for {ticker}: {str(e)}")

    print(f"\n  ✅ Fetched data for {len(market_data_map)}/{len(tickers)} stocks")
    print()

    return {"market_data": market_data_map}


def fix_json_errors(state: AgentState) -> Dict:
    """
    Repair malformed JSON responses using a specialized LLM prompt

    Args:
        state: Current agent state

    Returns:
        Updated state with repaired analyses added to video_analyses
    """
    print("🔧 REPAIRING JSON ERRORS")
    print("-" * 60)

    failed_parses = state.get('failed_json_parses', [])

    if not failed_parses:
        print("  ℹ️  No JSON errors to repair")
        print()
        return {}

    config = state['config']
    repaired_analyses = []
    repair_failures = []

    # Get LLM configuration (use same as main analyzer or fallback to anthropic)
    llm_provider = config.get('llm_provider', 'anthropic')
    api_key = None

    import os
    if llm_provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
    elif llm_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
    elif llm_provider == 'gemini':
        api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print(f"  ⚠️  {llm_provider.upper()}_API_KEY not set for JSON repair")
        print("  Skipping JSON repair - these videos will be marked as errors")
        for failed in failed_parses:
            state['errors'].append(f"JSON repair skipped (no API key): {failed['video_info']['title']}")
        print()
        return {}

    # Create LLM client
    try:
        if llm_provider == 'anthropic':
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            model = "claude-3-5-sonnet-20241022"
        elif llm_provider == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model = "gpt-4o"
        elif llm_provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel("gemini-2.0-flash")
            model = "gemini-2.0-flash"
    except ImportError as e:
        print(f"  ⚠️  Failed to import {llm_provider}: {str(e)}")
        print()
        return {}

    # Repair each failed JSON
    for i, failed in enumerate(failed_parses, 1):
        video_info = failed['video_info']
        raw_response = failed.get('raw_response')

        print(f"\n  [{i}/{len(failed_parses)}] Repairing: {video_info['title']}")

        if not raw_response:
            print(f"    ⚠️  No raw response available, cannot repair")
            repair_failures.append(failed)
            continue

        # Call LLM to repair JSON
        try:
            repaired_json = _repair_json_with_llm(
                client, llm_provider, model, raw_response
            )

            # Try to parse the repaired JSON
            import json
            analysis = json.loads(repaired_json)

            # Add video metadata
            analysis['video_url'] = video_info['url']
            analysis['channel'] = video_info['channel']
            analysis['title'] = video_info.get('title', 'Unknown')
            analysis['published_at'] = video_info.get('published_at', '')
            analysis['analysis_method'] = f"{failed['analyzer_type']}-repaired"

            repaired_analyses.append(analysis)
            print(f"    ✅ Successfully repaired - found {len(analysis.get('stocks', []))} stocks")

        except Exception as e:
            print(f"    ❌ Repair failed: {str(e)}")
            repair_failures.append(failed)

    # Add repair failures to errors
    for failed in repair_failures:
        error_msg = f"JSON repair failed for {failed['video_info']['title']}"
        state['errors'].append(error_msg)

    print(f"\n  ✅ Repaired: {len(repaired_analyses)}")
    print(f"  ❌ Failed: {len(repair_failures)}")
    print()

    # Merge repaired analyses with existing ones
    all_analyses = state['video_analyses'] + repaired_analyses

    return {
        "video_analyses": all_analyses,
        "failed_json_parses": []  # Clear the list since we've processed them
    }


def _repair_json_with_llm(client, llm_provider: str, model: str, malformed_json: str) -> str:
    """
    Use LLM to repair malformed JSON

    Args:
        client: LLM client instance
        llm_provider: Provider name
        model: Model name
        malformed_json: The malformed JSON string

    Returns:
        Repaired JSON string
    """
    import json
    import re

    prompt = f"""You are a JSON repair specialist. Your task is to fix malformed JSON.

MALFORMED JSON:
{malformed_json}

EXPECTED SCHEMA:
{{
    "main_thesis": "string",
    "stocks": [
        {{
            "ticker": "string",
            "company_name": "string",
            "action": "buy" | "sell" | "hold",
            "reasoning": "string",
            "confidence": 0.0 to 1.0,
            "catalysts": ["string"],
            "price_target": "string (optional)",
            "timeframe": "string (optional)"
        }}
    ]
}}

Common issues to fix:
1. Missing closing brackets/braces
2. Trailing commas before closing brackets
3. Incomplete stock objects in the array
4. Missing quotes around strings
5. Unclosed strings
6. Invalid escape sequences

Return ONLY the repaired, valid JSON. Do not include any explanation or markdown.
"""

    try:
        if llm_provider == 'anthropic':
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            # Remove markdown if present
            text = re.sub(r'```json\s*|\s*```', '', text).strip()
            return text

        elif llm_provider == 'openai':
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = response.choices[0].message.content
            text = re.sub(r'```json\s*|\s*```', '', text).strip()
            return text

        elif llm_provider == 'gemini':
            response = client.generate_content(prompt)
            text = response.text
            text = re.sub(r'```json\s*|\s*```', '', text).strip()
            return text

    except Exception as e:
        raise Exception(f"LLM repair failed: {str(e)}")


def critic_review(state: AgentState) -> Dict:
    """
    Review aggregated stocks for major inconsistencies

    Args:
        state: Current agent state

    Returns:
        Updated state with critic results
    """
    print("🔍 CRITIC REVIEW")
    print("-" * 60)

    config = state['config']
    critic_config = config.get('critic', {})

    # Check if critic is enabled
    if not critic_config.get('enabled', True):
        print("  ⏭️  Critic disabled, skipping review")
        return {
            "critic_result": {
                'critical_analysis': {},
                'summary': 'Critic disabled',
                'total_reviewed': len(state['aggregated_stocks'])
            }
        }

    # Import and create critic
    try:
        from .critic import StockCritic
        import os

        # Get LLM provider and API key
        llm_provider = critic_config.get('llm_provider', 'anthropic')
        model = critic_config.get('model')

        # Get API key (not needed for ollama)
        api_key = None
        if llm_provider == 'ollama':
            # Ollama doesn't need API key, use base URL from config
            ollama_base_url = config.get('ollama_base_url', 'http://localhost:11434')
            critic = StockCritic(
                llm_provider=llm_provider,
                model=model or config.get('ollama_model', 'qwen2.5:3b'),
                ollama_base_url=ollama_base_url
            )
        else:
            # Cloud providers need API key
            if llm_provider == 'anthropic':
                api_key = os.getenv('ANTHROPIC_API_KEY')
            elif llm_provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')
            elif llm_provider == 'gemini':
                api_key = os.getenv('GEMINI_API_KEY')

            if not api_key:
                print(f"  ⚠️  {llm_provider.upper()}_API_KEY not set, skipping critic review")
                return {
                    "critic_result": {
                        'critical_analysis': {},
                        'summary': f'{llm_provider} API key not available',
                        'total_reviewed': len(state['aggregated_stocks'])
                    }
                }

            # Create critic with API key
            critic = StockCritic(llm_provider=llm_provider, api_key=api_key, model=model)

        if not critic.is_available():
            print(f"  ⚠️  Critic not available, skipping review")
            return {
                "critic_result": {
                    'critical_analysis': {},
                    'summary': 'Critic not available',
                    'total_reviewed': len(state['aggregated_stocks'])
                }
            }

        # Run critic review with market data
        market_data = state.get('market_data', {})
        result = critic.review_stocks(
            state['aggregated_stocks'],
            state['video_analyses'],
            market_data=market_data if market_data else None
        )

        # Merge critical analysis into aggregated stocks
        critical_analysis = result.get('critical_analysis', {})
        aggregated_stocks = state['aggregated_stocks']

        for stock in aggregated_stocks:
            ticker = stock['ticker']
            if ticker in critical_analysis:
                stock['critical_analysis'] = critical_analysis[ticker]

        print()
        return {
            "critic_result": result,
            "aggregated_stocks": aggregated_stocks
        }

    except Exception as e:
        error_msg = f"Critic review failed: {str(e)}"
        print(f"  ❌ {error_msg}")
        return {
            "critic_result": {
                'critical_analysis': {},
                'summary': error_msg,
                'total_reviewed': len(state['aggregated_stocks'])
            }
        }


def deep_investigation(state: AgentState) -> Dict:
    """
    Investigate flagged stocks with real market data

    Args:
        state: Current agent state

    Returns:
        Updated state with investigation results
    """
    print("🔬 DEEP INVESTIGATION")
    print("-" * 60)

    config = state['config']
    investigator_config = config.get('investigator', {})
    flagged_stocks = state['critic_result'].get('flagged_stocks', [])

    if not flagged_stocks:
        print("  ℹ️  No stocks flagged for investigation")
        print()
        return {"investigation_results": []}

    # Check if investigator is enabled
    if not investigator_config.get('enabled', True):
        print("  ⏭️  Investigator disabled, skipping")
        print()
        return {"investigation_results": []}

    # Import and create investigator
    try:
        from .investigator import MarketInvestigator
        import os

        # Get LLM provider and API key
        llm_provider = investigator_config.get('llm_provider', 'anthropic')
        model = investigator_config.get('model')

        # Get API key (not needed for ollama)
        api_key = None
        if llm_provider == 'ollama':
            # Ollama doesn't need API key, use base URL from config
            ollama_base_url = config.get('ollama_base_url', 'http://localhost:11434')
            investigator = MarketInvestigator(
                llm_provider=llm_provider,
                model=model or config.get('ollama_model', 'qwen2.5:3b'),
                ollama_base_url=ollama_base_url
            )
        else:
            # Cloud providers need API key
            if llm_provider == 'anthropic':
                api_key = os.getenv('ANTHROPIC_API_KEY')
            elif llm_provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')
            elif llm_provider == 'gemini':
                api_key = os.getenv('GEMINI_API_KEY')

            if not api_key:
                print(f"  ⚠️  {llm_provider.upper()}_API_KEY not set, skipping investigation")
                print()
                return {"investigation_results": []}

            # Create investigator with API key
            investigator = MarketInvestigator(llm_provider=llm_provider, api_key=api_key, model=model)

        if not investigator.is_available():
            print(f"  ⚠️  Investigator not available (yfinance may not be installed)")
            print()
            return {"investigation_results": []}

        # Investigate each flagged stock
        max_investigations = investigator_config.get('max_investigations_per_run', 10)
        investigations = []

        for i, flagged in enumerate(flagged_stocks[:max_investigations], 1):
            ticker = flagged['ticker']
            concern = flagged['concern']
            investigation_focus = flagged.get('investigation_focus', 'general')

            # Find original stock data
            stock_data = next(
                (s for s in state['aggregated_stocks'] if s['ticker'] == ticker),
                None
            )

            if not stock_data:
                print(f"  ⚠️  Could not find data for {ticker}")
                continue

            print(f"\n  [{i}/{min(len(flagged_stocks), max_investigations)}] {ticker}")

            # Investigate
            result = investigator.investigate_stock(
                ticker=ticker,
                concern=concern,
                investigation_focus=investigation_focus,
                stock_data=stock_data
            )

            investigations.append(result)

        print()
        return {"investigation_results": investigations}

    except Exception as e:
        error_msg = f"Investigation failed: {str(e)}"
        print(f"  ❌ {error_msg}")
        print()
        return {"investigation_results": []}


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
    from datetime import datetime
    report.append("# 📊 Investment Analysis Report\n")
    report.append(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n")
    report.append(f"*Analyzed {len(state['video_analyses'])} videos across {len(state['channels'])} channels*\n")
    report.append(f"*Total cost: ${state['total_cost']:.2f}*\n")
    report.append("---\n")

    # Critic review section
    critic_result = state.get('critic_result', {})
    if critic_result and critic_result.get('total_reviewed', 0) > 0:
        report.append("## 🔍 Critical Analysis Summary\n")
        report.append(f"*{critic_result.get('summary', '')}*\n")
        report.append("---\n")

    # Investigation results section
    investigations = state.get('investigation_results', [])
    if investigations:
        report.append("## 🔬 Deep Investigation Results\n")
        report.append(f"*{len(investigations)} stock(s) investigated with real market data*\n")

        for inv in investigations:
            ticker = inv['ticker']
            verdict = inv['verdict']
            reasoning = inv['reasoning']
            recommendation = inv.get('recommendation', 'caution')
            market_data = inv.get('market_data', {})

            # Verdict emoji
            verdict_emoji = {
                'concern_dismissed': '✅',
                'concern_valid': '❌',
                'inconclusive': '❓'
            }.get(verdict, '❓')

            report.append(f"\n### {verdict_emoji} {ticker} - {verdict.replace('_', ' ').title()}\n")

            # Market data summary
            if market_data and market_data.get('current_price'):
                report.append(f"**Market Data:**")
                report.append(f"\n- Current Price: ${market_data['current_price']:.2f}")

                if market_data.get('pe_ratio'):
                    report.append(f"\n- P/E Ratio: {market_data['pe_ratio']:.2f}")

                if market_data.get('analyst_target_price'):
                    report.append(f"\n- Analyst Target: ${market_data['analyst_target_price']:.2f}")

                if market_data.get('month_performance') is not None:
                    report.append(f"\n- 1-Month Performance: {market_data['month_performance']:+.2f}%")

                report.append("\n")

            # Verdict
            report.append(f"\n**Verdict:** {reasoning}\n")
            report.append(f"**Recommendation:** {recommendation.upper()}\n")

            # Key findings
            if inv.get('key_findings'):
                report.append(f"\n**Key Findings:**")
                for finding in inv['key_findings']:
                    report.append(f"\n- {finding}")
                report.append("\n")

        report.append("---\n")

    # Historical context section
    historical_insights = _generate_historical_insights(state)
    if historical_insights:
        report.append("## 📚 Historical Context & Trends\n")
        report.append(historical_insights)
        report.append("\n---\n")

    # High priority stocks (multiple channels)
    multi_channel = [s for s in state['aggregated_stocks'] if s['num_channels'] > 1]

    if multi_channel:
        report.append("## 🔥 HIGH PRIORITY (Multiple Channel Coverage)\n")

        for stock in multi_channel:
            # Stock header
            ticker = stock['ticker']
            report.append(f"### {ticker} - {stock['company_name']}\n")
            report.append(f"**Coverage:** {stock['num_channels']} channels | ")
            report.append(f"**Sentiment:** {'✅ Aligned' if stock['sentiment_aligned'] else '⚠️ Mixed'} ({stock['dominant_action'].upper()}) | ")
            report.append(f"**Avg Confidence:** {stock['avg_confidence']:.0%}\n")

            # Add market data if available
            market_data_dict = state.get('market_data', {})
            if ticker in market_data_dict:
                mdata = market_data_dict[ticker]
                report.append("\n**📊 Market Data:**")

                if mdata.get('current_price'):
                    report.append(f"\n- Current Price: ${mdata['current_price']:.2f}")

                if mdata.get('pe_ratio'):
                    report.append(f"\n- P/E Ratio: {mdata['pe_ratio']:.2f}")

                if mdata.get('analyst_target_price'):
                    report.append(f"\n- Analyst Avg Target: ${mdata['analyst_target_price']:.2f}")

                    # Show upside/downside
                    if mdata.get('current_price'):
                        upside = ((mdata['analyst_target_price'] - mdata['current_price']) / mdata['current_price']) * 100
                        report.append(f" ({upside:+.1f}% upside)")

                if mdata.get('recommendation_mean'):
                    rec_map = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Sell", 5: "Strong Sell"}
                    rec_value = round(mdata['recommendation_mean'])
                    report.append(f"\n- Analyst Consensus: {rec_map.get(rec_value, 'N/A')}")

                if mdata.get('month_performance') is not None:
                    report.append(f"\n- 1-Month Performance: {mdata['month_performance']:+.2f}%")

                report.append("\n")

            # Per-channel details
            for mention in stock['mentions']:
                report.append(f"\n**{mention['channel']}** - [{mention['video_title'][:60]}...]({mention['video_url']})")
                report.append(f"- **Action:** {mention['action'].upper()}")
                report.append(f"- **Reasoning:** {mention['reasoning']}")
                report.append(f"- **Confidence:** {mention['confidence']:.0%}")

                # Show thesis evolution if present (this is the key insight!)
                if mention.get('thesis_evolution'):
                    report.append(f"\n  **📈 Thesis Evolution:**")
                    report.append(f"  > {mention['thesis_evolution']}\n")
                elif mention.get('historical_note'):
                    report.append(f"- **📊 Historical Note:** {mention['historical_note']}")

                if mention['catalysts']:
                    report.append(f"- **Catalysts:** {', '.join(mention['catalysts'])}")

                if mention['price_target']:
                    report.append(f"- **Price Target:** {mention['price_target']}")

                if mention['timeframe']:
                    report.append(f"- **Timeframe:** {mention['timeframe']}")

                report.append("")

            # Add critical analysis if available
            if stock.get('critical_analysis'):
                crit = stock['critical_analysis']
                report.append("\n**⚖️ Critical Analysis:**")

                if crit.get('alternative_view'):
                    report.append(f"- **Alternative Perspective:** {crit['alternative_view']}")

                if crit.get('counterpoints'):
                    report.append(f"- **Counterpoints:**")
                    for cp in crit['counterpoints']:
                        report.append(f"  - {cp}")

                if crit.get('risks'):
                    report.append(f"- **Key Risks:**")
                    for risk in crit['risks']:
                        report.append(f"  - {risk}")

                if crit.get('red_flags'):
                    report.append(f"- **⚠️ Red Flags:**")
                    for flag in crit['red_flags']:
                        report.append(f"  - {flag}")

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

            # Add critical analysis if available (brief version for single channel)
            if stock.get('critical_analysis'):
                crit = stock['critical_analysis']
                if crit.get('risks') and len(crit['risks']) > 0:
                    report.append(f"\n  - ⚠️ Risk: {crit['risks'][0]}")

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


def _generate_historical_insights(state: AgentState) -> str:
    """
    Generate historical insights section from video analyses

    Args:
        state: Current agent state

    Returns:
        Formatted historical insights string
    """
    insights = []

    # Check if any video has historical notes or thesis evolution
    stocks_with_evolution = {}
    stocks_with_history = {}

    for analysis in state['video_analyses']:
        for stock in analysis.get('stocks', []):
            ticker = stock.get('ticker', '').upper()
            if not ticker:
                continue

            # Prioritize thesis evolution over simple historical notes
            thesis_evolution = stock.get('thesis_evolution', '')
            historical_note = stock.get('historical_note', '')

            if thesis_evolution:
                if ticker not in stocks_with_evolution:
                    stocks_with_evolution[ticker] = []

                stocks_with_evolution[ticker].append({
                    'channel': analysis.get('channel', 'Unknown'),
                    'evolution': thesis_evolution,
                    'action': stock.get('action', 'unknown'),
                    'confidence': stock.get('confidence', 0.0)
                })
            elif historical_note:
                if ticker not in stocks_with_history:
                    stocks_with_history[ticker] = []

                stocks_with_history[ticker].append({
                    'channel': analysis.get('channel', 'Unknown'),
                    'note': historical_note,
                    'action': stock.get('action', 'unknown'),
                    'confidence': stock.get('confidence', 0.0)
                })

    if not stocks_with_evolution and not stocks_with_history:
        return ""

    # Show thesis evolutions first (more important)
    if stocks_with_evolution:
        insights.append("### 📈 Significant Thesis Changes\n")
        insights.append("*These stocks have evolved investment theses based on new information:*\n")

        for ticker, evolutions in sorted(stocks_with_evolution.items()):
            insights.append(f"\n**{ticker}**")
            for evo_data in evolutions:
                insights.append(f"- *{evo_data['channel']}* (Current: {evo_data['action'].upper()}, Confidence: {evo_data['confidence']:.0%})")
                insights.append(f"  > {evo_data['evolution']}")

    # Then show recurring mentions
    if stocks_with_history:
        if stocks_with_evolution:
            insights.append("\n")
        insights.append("### 🔁 Recurring Mentions\n")
        insights.append("*These stocks were mentioned in previous analyses:*\n")

        for ticker, notes in sorted(stocks_with_history.items()):
            insights.append(f"\n**{ticker}**")
            for note_data in notes:
                insights.append(
                    f"- *{note_data['channel']}*: {note_data['note']} "
                    f"(Current: {note_data['action'].upper()}, "
                    f"Confidence: {note_data['confidence']:.0%})"
                )

    return "\n".join(insights)


def should_repair_json(state: AgentState) -> str:
    """
    Decide whether to repair JSON errors before aggregation

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    failed_json = state.get('failed_json_parses', [])

    if failed_json:
        return "fix_json_errors"
    else:
        return "aggregate_stocks"


def should_investigate(state: AgentState) -> str:
    """
    Decide whether to run deep investigation based on critic results

    Note: Since critic now provides critical analysis instead of flagging stocks,
    investigation is skipped by default. This function is kept for backwards compatibility.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    # Critic no longer flags stocks for investigation
    # It provides critical analysis inline instead
    return "generate_report"


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
    workflow.add_node("fix_json_errors", fix_json_errors)
    workflow.add_node("aggregate_stocks", aggregate_stocks)
    workflow.add_node("fetch_market_data", fetch_market_data)
    workflow.add_node("critic_review", critic_review)
    workflow.add_node("deep_investigation", deep_investigation)
    workflow.add_node("generate_report", generate_report)

    # Define edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "collect_videos")
    workflow.add_edge("collect_videos", "analyze_videos")

    # Conditional edge: repair JSON if there are parsing errors
    workflow.add_conditional_edges(
        "analyze_videos",
        should_repair_json,
        {
            "fix_json_errors": "fix_json_errors",
            "aggregate_stocks": "aggregate_stocks"
        }
    )

    # After JSON repair, go to aggregation
    workflow.add_edge("fix_json_errors", "aggregate_stocks")

    # After aggregation, fetch market data
    workflow.add_edge("aggregate_stocks", "fetch_market_data")

    # After market data, run critic review
    workflow.add_edge("fetch_market_data", "critic_review")

    # Conditional edge: investigate if stocks are flagged
    workflow.add_conditional_edges(
        "critic_review",
        should_investigate,
        {
            "deep_investigation": "deep_investigation",
            "generate_report": "generate_report"
        }
    )

    workflow.add_edge("deep_investigation", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()
