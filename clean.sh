#!/usr/bin/env bash

set -euo pipefail

# Clean up Python bytecode files and caches (quiet)
find . -type f -name "*.pyc" -exec rm -f {} \;
find . -type d -name "__pycache__" -exec rm -rf {} \;
find . -type f -name "*.pyo" -exec rm -f {} \;
find . -type f -name "*~" -exec rm -f {} \;
find . -type f -name ".DS_Store" -exec rm -f {} \;
find . -type d -name ".mypy*" -exec rm -rf {} \;
find . -type d -name ".pytest_cache" -exec rm -rf {} \;
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} \;
find . -type d -name ".vscode" -exec rm -rf {} \;
find . -type d -name ".idea" -exec rm -rf {} \;
find . -type d -name "__pypackages__" -exec rm -rf {} \;
find . -type d -name "build" -exec rm -rf {} \;
find . -type d -name "dist" -exec rm -rf {} \;
find . -type d -name "*.egg-info" -exec rm -rf {} \;
find . -type d -name ".eggs" -exec rm -rf {} \;
find . -type d -name ".tox" -exec rm -rf {} \;
find . -type d -name ".mypy_cache" -exec rm -rf {} \;

echo "Cleaned up Python bytecode files and caches."

# Optionally remove virtual environments
# find . -type d -name "venv"   -exec rm -rf {} \;
# find . -type d -name ".venv"  -exec rm -rf {} \;
# find . -type d -name "env"    -exec rm -rf {} \;
# find . -type d -name ".env"   -exec rm -rf {} \;
# echo "Removed virtual environment directories."

echo "Cleanup complete."
