#!/bin/bash
# Setup Pre-commit Hooks
# This script installs and configures pre-commit hooks for the project

echo "🔧 Setting up pre-commit hooks..."
echo "================================"

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "❌ pre-commit is not installed"
    echo "📦 Installing pre-commit..."
    pip install pre-commit
fi

# Install the git hooks
echo "📌 Installing git hooks..."
pre-commit install

# Optional: Run against all files to verify
echo ""
echo "✅ Pre-commit hooks installed successfully!"
echo ""
echo "💡 To test hooks on all files, run:"
echo "   pre-commit run --all-files"
echo ""
echo "📝 Hooks that will run on every commit:"
echo "   - trailing-whitespace: Remove trailing whitespace"
echo "   - end-of-file-fixer: Ensure files end with newline"
echo "   - check-yaml: Validate YAML syntax"
echo "   - check-json: Validate JSON syntax"
echo "   - check-merge-conflict: Detect merge conflict strings"
echo "   - detect-private-key: Prevent committing private keys"
echo "   - black: Format Python code"
echo "   - flake8: Lint Python code"
echo "   - isort: Sort Python imports"
