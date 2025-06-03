#!/bin/bash
set -e

echo "🔧 Creating virtual environment..."
python3 -m venv .venv

echo "📦 Activating environment and installing package..."
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
pip install ipykernel

echo "🧠 Registering Jupyter kernel..."
python -m ipykernel install --user --name fogo-kernel --display-name "Python (Fogo)"

echo "✅ Setup complete. Use the 'Python (Fogo)' kernel in your notebooks."