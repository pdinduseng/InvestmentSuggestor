"""Helper utility functions"""

import re
import json
from typing import Dict


def extract_video_id(url: str) -> str:
    """
    Extract video ID from YouTube URL

    Supports various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - VIDEO_ID (direct ID)

    Args:
        url: YouTube URL or video ID

    Returns:
        Video ID

    Raises:
        ValueError: If video ID cannot be extracted
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract video ID from: {url}")


def parse_json_response(text: str) -> Dict:
    """
    Parse JSON from LLM response

    Handles common issues like markdown code blocks

    Args:
        text: Raw response text

    Returns:
        Parsed dictionary

    Raises:
        json.JSONDecodeError: If JSON cannot be parsed
    """
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*|\s*```', '', text).strip()

    # Try to extract JSON if wrapped in other text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    return json.loads(text)


def format_stock_action(action: str) -> str:
    """
    Normalize stock action strings

    Args:
        action: Raw action string

    Returns:
        Normalized action ('buy', 'sell', or 'hold')
    """
    action = action.lower().strip()

    buy_synonyms = ['buy', 'long', 'bullish', 'accumulate', 'strong buy']
    sell_synonyms = ['sell', 'short', 'bearish', 'reduce', 'strong sell']
    hold_synonyms = ['hold', 'neutral', 'wait', 'watch']

    if any(syn in action for syn in buy_synonyms):
        return 'buy'
    elif any(syn in action for syn in sell_synonyms):
        return 'sell'
    elif any(syn in action for syn in hold_synonyms):
        return 'hold'
    else:
        return action  # Return as-is if can't normalize


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to maximum length

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + '...'
