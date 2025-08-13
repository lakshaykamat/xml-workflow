#!/bin/bash

# LLM LoRA Project Setup Script
# This script sets up the Python environment for the project

echo "🚀 Setting up LLM LoRA Project..."

# Check if Python 3.10.12 is available
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10.12 is not installed!"
    echo "Please install Python 3.10.12 first:"
    echo "  - Using pyenv: pyenv install 3.10.12"
    echo "  - Using conda: conda create -n llm-lora python=3.10.12"
    echo "  - Or download from python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3.10 --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" != "3.10.12" ]]; then
    echo "⚠️  Warning: Found Python $PYTHON_VERSION, but 3.10.12 is recommended"
    echo "Continuing anyway..."
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.10 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "You can now run:"
echo "  python src/train_lora.py  # for training"
echo "  python src/main.py        # for inference"
