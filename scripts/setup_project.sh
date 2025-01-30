#!/bin/bash

# 🚀 Project Setup Script using UV and setup.py
# This script sets up the project, installs uv, pins the Python version,
# creates a virtual environment, installs dependencies, and ensures setup.py exists.

# ========================
# 🔧 Configurable Variables
# ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$ROOT_DIR" || { log "❌ Failed to enter project root directory: $ROOT_DIR"; exit 1; }

PROJECT_NAME="complexity"  # Project name
PYTHON_DIR="server/src"   # Directory containing Python code
PYTHON_VERSION="3.10.16"  # Exact Python version to use

# ========================
# 🛠️ Utility Functions
# ========================
log() {
    echo -e "\n[$(date +"%Y-%m-%d %H:%M:%S")] $1"
}

create_setup_py() {
    if [[ ! -f "$ROOT_DIR/setup.py" ]]; then
        log "📄 Creating setup.py..."
        cat << EOF > "$ROOT_DIR/setup.py"
from setuptools import setup, find_packages

setup(
    name="$PROJECT_NAME",
    version="0.1.0",
    description="Complexity classification system",
    author="Your Name",
    author_email="your.email@example.com",
    python_requires=">=3.10",
    packages=find_packages(where="server/src"),
    package_dir={"": "server/src"},
    install_requires=[
        "datasets",
        "pandas",
        "matplotlib",
        "seaborn",
        "python-dotenv",
        "litellm",
        "requests",
        "markdownify",
        "tenacity>=9.0.0",
        "loguru>=0.7.3",
        "spacy>=3.8.4",
        "python-arango>=8.1.4",
        "deepmerge>=2.0",
        "torch>=2.2.2",
        "transformers>=4.48.1",
        "sentence-transformers>=3.4.0",
        "nltk>=3.9.1",
        "pyperclip>=1.9.0",
        "json-repair>=0.35.0",
        "rapidfuzz>=3.11.0"
    ],
    extras_require={
        "dev": [
            "pytest",
            "ipython",
            "jupyterlab",
            "pre-commit>=3.6.0"
        ]
    },
)
EOF
        log "✅ setup.py created."
    else
        log "✅ setup.py already exists."
    fi
}

install_package() {
    log "📦 Installing project in editable mode using setup.py..."
    uv pip install -e . || { log "❌ Failed to install project in editable mode"; exit 1; }
    log "✅ Project installed in editable mode."
}

# ========================
# 🚀 Main Script Logic
# ========================

log "🔍 Checking for UV installation..."
if ! command -v uv &> /dev/null; then
    log "🚀 UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || { log "❌ Failed to install UV"; exit 1; }
else
    log "✅ UV is already installed."
fi

# Clean up existing environment if it exists
if [[ -d "$ROOT_DIR/.venv" ]]; then
    log "🧹 Cleaning up existing virtual environment..."
    rm -rf "$ROOT_DIR/.venv"
fi

log "🐍 Ensuring Python $PYTHON_VERSION is available..."
if ! uv python list --all | grep -q "$PYTHON_VERSION"; then
    log "📥 Python $PYTHON_VERSION not found in UV's distribution list. Installing..."
    uv python install "$PYTHON_VERSION" || { 
        log "❌ Failed to install Python version $PYTHON_VERSION";
        log "💡 Available versions: $(uv python list --all)";
        exit 1; 
    }
fi

uv python pin "$PYTHON_VERSION" || { 
    log "❌ Failed to pin Python version $PYTHON_VERSION";
    exit 1; 
}

log "🔄 Creating new virtual environment..."
uv venv || { log "❌ Failed to create virtual environment"; exit 1; }

log "🔍 Verifying Python version in virtual environment..."
source "$ROOT_DIR/.venv/bin/activate"
ACTIVE_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$ACTIVE_VERSION" != "$PYTHON_VERSION" ]]; then
    log "❌ Virtual environment is using Python $ACTIVE_VERSION instead of $PYTHON_VERSION";
    exit 1;
else
    log "✅ Successfully using Python $PYTHON_VERSION";
fi

log "📂 Navigating to project root..."
cd "$ROOT_DIR" || { log "❌ Failed to enter project root directory: $ROOT_DIR"; exit 1; }

log "📦 Initializing UV project..."
uv init || { log "❌ Failed to initialize UV project"; exit 1; }

create_setup_py

log "📦 Installing dependencies..."
install_package

log "🎉 Project setup complete!"
echo -e "\nTo start working on your project:"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Add your Python code to the '$PYTHON_DIR' directory"
echo "3. Run your code using: python $PYTHON_DIR/your_script.py"

exit 0
