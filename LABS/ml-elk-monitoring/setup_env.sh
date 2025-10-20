#!/bin/bash

echo "Setting up Python virtual environment for ML ELK Monitoring..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip and setup tools..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Virtual environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo ""
echo "Installed packages:"
pip list

# Generate some initial logs
echo ""
echo "Generating initial ML logs..."
python scripts/ml_log_generator.py --batch 100

echo ""
echo "Setup complete! You can now:"
echo "1. Access Kibana at http://localhost:5601"
echo "2. Generate more logs: python scripts/ml_log_generator.py --batch 500"
echo "3. Generate continuous logs: python scripts/ml_log_generator.py --continuous --rate 20"
