#!/bin/bash

# 🚀 Project Setup Script using UV
# This script sets up the project, installs uv, pins the Python version,
# creates a virtual environment, installs dependencies, and runs a test.

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

create_gitignore() {
    log "📝 Creating .gitignore..."
    cat << EOF > "$ROOT_DIR/.gitignore"
# Standard ignores
.venv/
.vscode/
.env
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Data/logs
data/
logs/
*.csv
*.tsv
*.jsonl

# Experimental
experiments/
results/
notebooks/

# Large files
*.pt
*.bin
*.h5
*.zip
*.tar.gz

# Local development
.DS_Store
.idea/
EOF
    log "✅ .gitignore created."
}

create_env_file() {
    log "📝 Creating .env file..."
    cat << EOF > "$ROOT_DIR/.env"
# Set Python path to include the project's source directory
PYTHONPATH=$ROOT_DIR/$PYTHON_DIR

# Hugging Face configurations
HF_TOKEN=your_huggingface_token_here
HF_HUB_ENABLE_HF_TRANSFER=True  # Enable high-speed downloads
HF_HUB_DOWNLOAD_TIMEOUT=300     # Increase timeout for large files

# Add other environment variables here
EOF
    log "✅ .env file created."
}

check_env_vars() {
    log "🔍 Checking required environment variables..."
    if [[ -z "$HF_TOKEN" ]]; then
        log "❌ HF_TOKEN is not set in .env file";
        log "💡 Add your Hugging Face token to the .env file";
        exit 1;
    else
        log "✅ HF_TOKEN is set.";
    fi
}

ensure_python_package() {
    log "📂 Ensuring '$PYTHON_DIR' is a Python package..."
    mkdir -p "$ROOT_DIR/$PYTHON_DIR"
    touch "$ROOT_DIR/$PYTHON_DIR/__init__.py"
    log "✅ '$PYTHON_DIR' is now a Python package."
}

install_dependencies() {
    if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
        log "📄 Found requirements.txt. Installing dependencies from it..."
        uv pip sync requirements.txt || { log "❌ Failed to install dependencies from requirements.txt"; exit 1; }
        log "✅ Dependencies installed from requirements.txt."
    else
        log "📄 No requirements.txt found. Syncing dependencies from pyproject.toml..."
        uv pip sync || { log "❌ Failed to sync dependencies from pyproject.toml"; exit 1; }
        log "✅ Dependencies synced from pyproject.toml."
    fi
}

run_module_test() {
    log "🧪 Running module test..."
    source "$ROOT_DIR/.venv/bin/activate" || { log "❌ Failed to activate virtual environment"; exit 1; }
    python -c "import smolagent; print('✅ SmolAgent module imported successfully!')" || {
        log "❌ Module test failed. Ensure 'smolagent' is properly configured.";
        exit 1;
    }
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
# Verify version exists in UV's list
if ! uv python list --all | grep -q "$PYTHON_VERSION"; then
    log "📥 Python $PYTHON_VERSION not found in UV's distribution list. Installing..."
    uv python install "$PYTHON_VERSION" || { 
        log "❌ Failed to install Python version $PYTHON_VERSION";
        log "💡 Available versions: $(uv python list --all)";
        exit 1; 
    }
fi

# Pin and verify version
uv python pin "$PYTHON_VERSION" || { 
    log "❌ Failed to pin Python version $PYTHON_VERSION";
    exit 1; 
}

# Create fresh virtual environment
log "🔄 Creating new virtual environment..."
uv venv || { log "❌ Failed to create virtual environment"; exit 1; }

# Verify active Python version
log "🔍 Verifying Python version in virtual environment..."
source "$ROOT_DIR/.venv/bin/activate"
ACTIVE_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$ACTIVE_VERSION" != "$PYTHON_VERSION" ]]; then
    log "❌ Critical error: Virtual environment is using Python $ACTIVE_VERSION instead of $PYTHON_VERSION";
    log "💡 Try these steps:";
    log "1. Delete the .venv directory: rm -rf .venv";
    log "2. Verify Python installation: uv python list --all";
    log "3. Run the setup script again";
    exit 1;
else
    log "✅ Successfully using Python $PYTHON_VERSION";
fi

log "📂 Navigating to project root..."
cd "$ROOT_DIR" || { log "❌ Failed to enter project root directory: $ROOT_DIR"; exit 1; }

log "📦 Initializing UV project..."
uv init || { log "❌ Failed to initialize UV project"; exit 1; }

ensure_python_package

create_gitignore
create_env_file
check_env_vars

log "🔄 Creating virtual environment and syncing dependencies..."
uv venv || { log "❌ Failed to create virtual environment"; exit 1; }
uv pip sync || { log "❌ Failed to sync dependencies"; exit 1; }


install_dependencies

log "📦 Installing project in editable mode..."
uv pip install -e . || { log "❌ Failed to install project in editable mode"; exit 1; }

run_module_test

log "🎉 Project setup complete!"
echo -e "\nTo start working on your project:"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Add your Python code to the '$PYTHON_DIR' directory"
echo "3. Run your code using: python $PYTHON_DIR/your_script.py"

exit 0
