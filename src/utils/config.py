"""Configuration management utilities"""

import os
import yaml
from typing import Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file with environment variable substitution

    Args:
        config_path: Path to config.yaml file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please create a config.yaml file or use config.example.yaml as a template."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    config = _substitute_env_vars(config)

    # Validate configuration
    _validate_config(config)

    return config


def _substitute_env_vars(config: Dict) -> Dict:
    """
    Recursively substitute environment variables in config

    Supports ${VAR_NAME} syntax

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with substituted values
    """
    if isinstance(config, dict):
        return {key: _substitute_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        # Extract environment variable name
        env_var = config[2:-1]
        value = os.getenv(env_var)
        return value  # Can be None if not set
    else:
        return config


def _validate_config(config: Dict) -> None:
    """
    Validate configuration

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    if 'analysis_mode' not in config:
        raise ValueError("Configuration must include 'analysis_mode'")

    valid_modes = ['ollama', 'gemini', 'youtube_api', 'hybrid', 'auto']
    if config['analysis_mode'] not in valid_modes:
        raise ValueError(
            f"Invalid analysis_mode: {config['analysis_mode']}. "
            f"Must be one of: {', '.join(valid_modes)}"
        )

    if 'channels' not in config or not config['channels']:
        raise ValueError("Configuration must include at least one channel")

    # Validate channels
    for i, channel in enumerate(config['channels']):
        if 'channel_id' not in channel:
            raise ValueError(f"Channel {i} missing 'channel_id'")
        if 'name' not in channel:
            raise ValueError(f"Channel {i} missing 'name'")

    # Check that at least one API key is provided
    api_keys = [
        config.get('gemini_api_key'),
        config.get('openai_api_key'),
        config.get('anthropic_api_key')
    ]

    if not any(api_keys):
        print("⚠️ WARNING: No API keys found in configuration!")
        print("   Make sure to set environment variables:")
        print("   - GEMINI_API_KEY (for Gemini)")
        print("   - OPENAI_API_KEY (for OpenAI)")
        print("   - ANTHROPIC_API_KEY (for Anthropic)")
