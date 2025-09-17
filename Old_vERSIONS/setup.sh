#!/bin/bash

# Stock Analysis App Setup Script

echo "🚀 Setting up Stock Analysis Streamlit App..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv stock_analysis_env

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source stock_analysis_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing required packages..."
pip install -r requirements.txt

echo "✅ Installation complete!"
echo ""
echo "🎯 To run the app:"
echo "   1. Activate the environment: source stock_analysis_env/bin/activate"
echo "   2. Run the app: streamlit run streamlit_app.py"
echo ""
echo "🌐 The app will open in your default web browser"
echo "📊 Happy analyzing!"
