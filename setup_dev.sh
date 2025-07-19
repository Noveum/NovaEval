#!/bin/bash
# NovaEval Development Setup Script
# This script sets up the complete development environment for NovaEval

set -e  # Exit on any error

echo "🚀 Setting up NovaEval development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_section "Creating Virtual Environment"
    python -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_section "Activating Virtual Environment"
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
print_section "Upgrading pip"
pip install --upgrade pip

# Install development dependencies
print_section "Installing Development Dependencies"
pip install -e ".[dev]"
print_status "Development dependencies installed"

# Install pre-commit hooks
print_section "Setting up Pre-commit Hooks"
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_status "Pre-commit hooks installed"
else
    print_error "pre-commit not found. Installing..."
    pip install pre-commit
    pre-commit install
    print_status "Pre-commit installed and hooks configured"
fi

# Create .vscode directory if it doesn't exist
if [ ! -d ".vscode" ]; then
    mkdir -p .vscode
    print_status ".vscode directory created"
fi

# Set up environment variables
print_section "Setting up Environment Variables"
cat > .env << EOF
# NovaEval Development Environment Variables
NOVAEVAL_DEBUG=1
NOVAEVAL_LOG_LEVEL=DEBUG
PYTHONPATH=./src
EOF
print_status "Environment variables configured in .env file"

# Create a basic test run
print_section "Running Basic Tests"
if python -m pytest tests/ -v --tb=short -x; then
    print_status "Basic tests passed!"
else
    print_warning "Some tests failed. This is expected in development."
fi

# Create useful aliases
print_section "Setting up Development Aliases"
cat > dev_aliases.sh << 'EOF'
#!/bin/bash
# Development aliases for NovaEval

# Activate virtual environment
alias venv='source venv/bin/activate'

# Common commands
alias novaeval-test='python -m pytest tests/ -v'
alias novaeval-test-cov='python -m pytest tests/ --cov=novaeval --cov-report=html'
alias novaeval-lint='ruff check src/ tests/'
alias novaeval-format='black src/ tests/ && isort src/ tests/'
alias novaeval-typecheck='mypy src/novaeval'

# Debug commands
alias novaeval-debug='python debug_setup.py'
alias novaeval-debug-cli='python debug_setup.py cli'
alias novaeval-debug-eval='python debug_setup.py eval'
alias novaeval-debug-tests='python debug_setup.py tests'

# CLI shortcuts
alias neval='python -m novaeval.cli'
alias neval-help='python -m novaeval.cli --help'
alias neval-version='python -m novaeval.cli --version'

echo "NovaEval development aliases loaded!"
echo "Usage examples:"
echo "  venv                 - Activate virtual environment"
echo "  novaeval-test        - Run tests"
echo "  novaeval-debug-cli   - Debug CLI"
echo "  neval --help         - Show CLI help"
EOF

chmod +x dev_aliases.sh
print_status "Development aliases created in dev_aliases.sh"

print_section "Setup Complete! 🎉"
echo ""
echo "To get started:"
echo "1. Load development aliases: source dev_aliases.sh"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run tests: novaeval-test"
echo "4. Debug CLI: python debug_setup.py cli"
echo "5. Debug evaluation: python debug_setup.py eval"
echo ""
echo "In VS Code/Cursor:"
echo "- Use F5 to start debugging with the configurations in .vscode/launch.json"
echo "- Set breakpoints in your code and debug interactively"
echo ""
echo "Environment variables are set in .env file"
echo "Development aliases are in dev_aliases.sh"
echo ""
print_status "Happy coding! 🚀"
