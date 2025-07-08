#!/bin/bash

# Temporal Graph RAG Setup Script for Linux/macOS

set -e

echo "🏁 Temporal Graph RAG Setup"
echo "=========================="

# Check Python version
python3 --version || { echo "❌ Python 3 is required"; exit 1; }

# Create virtual environment
echo "🔄 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install package
echo "🔄 Installing Temporal Graph RAG..."
pip install -e .

# Install optional dependencies
echo "🔄 Installing optional dependencies..."
pip install faiss-cpu || echo "⚠️ FAISS installation failed (optional)"

# Create directories
echo "📁 Creating directories..."
mkdir -p models results logs

# Setup environment file
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "📝 Created .env file from example"
        echo "⚠️  Please edit .env with your API credentials"
    fi
fi

# Run tests
echo "🧪 Running tests..."
python tests/test_basic.py

echo ""
echo "✅ Setup completed!"
echo "==================="
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Edit .env with your credentials"
echo "3. Try: python -m temporal_graph_rag --help"
echo "4. Read QUICKSTART.md for examples"
