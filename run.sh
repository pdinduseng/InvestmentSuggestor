#!/bin/bash

# Investment Analysis Agent - Quick Run Script

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️ .env file not found. Using environment variables if available."
fi

# Run the agent
python main.py "$@"
