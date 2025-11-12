#!/bin/bash

# Investment Analysis Agent - Setup Script

echo "📊 Investment Analysis Agent - Setup"
echo "===================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Found Python $python_version"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"
echo ""

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please edit .env and add your API keys!"
else
    echo "✅ .env file already exists"
fi
echo ""

# Make main.py executable
chmod +x main.py

echo "===================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. (Optional) Edit config.yaml to customize channels"
echo "3. Run: source venv/bin/activate"
echo "4. Run: export \$(cat .env | xargs)"
echo "5. Run: python main.py"
echo ""
echo "Or use the quick start script:"
echo "  ./run.sh"
echo ""
