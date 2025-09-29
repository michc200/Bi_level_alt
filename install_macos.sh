#!/bin/bash

# install_macos.sh - Setup script for DSML project on macOS using uv
# This script creates a uv environment with all required dependencies to run test.py

set -e  # Exit on any error

echo "Starting DSML environment setup for macOS..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "Found uv: $(uv --version)"

# Create project directory if it doesn't exist
PROJECT_NAME="dsml-env"
echo "Creating uv project: $PROJECT_NAME"

# Remove existing environment if it exists
if [ -d "$PROJECT_NAME" ]; then
    echo "Removing existing environment: $PROJECT_NAME"
    rm -rf "$PROJECT_NAME"
fi

# Initialize uv project and install dependencies
uv init $PROJECT_NAME
cd $PROJECT_NAME

echo "Installing Python and dependencies..."

# Install the robusttest package which will pull in all dependencies from pyproject.toml
# Note: This includes plotly<5.0.0 to ensure compatibility with pandapower's titlefont usage
uv add --editable ../dsml-main/

# Navigate back to main directory
cd ..

echo ""
echo "Setup complete."
echo ""
echo "Usage:"
echo "  To run test.py:"
echo "    uv run --project dsml-env python dsml-main/examples/test.py"
echo ""
echo "  To run Python interactively:"
echo "    uv run --project dsml-env python"
echo ""
echo "  To run Jupyter:"
echo "    uv run --project dsml-env jupyter notebook dsml-main/examples/"
echo ""
echo "The environment is ready for DSML development."