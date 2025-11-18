#!/bin/bash

# WeatherFlow Project Setup Script
# This script initializes the complete project structure

set -e  # Exit on error

echo "=========================================="
echo "  WeatherFlow Project Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if project directory name is provided
PROJECT_NAME=${1:-weatherflow}

print_info "Creating project: $PROJECT_NAME"
echo ""

# Create main project directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

print_success "Created main project directory"

# Create directory structure
print_info "Creating directory structure..."

# Core directories
mkdir -p dags
mkdir -p src/{data,models,storage,utils}
mkdir -p tests
mkdir -p data/{raw,processed,features,predictions}
mkdir -p models
mkdir -p logs/airflow
mkdir -p config
mkdir -p notebooks
mkdir -p dashboard/components
mkdir -p scripts
mkdir -p docs

print_success "Directory structure created"

# Create __init__.py files
print_info "Creating Python package files..."

touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/storage/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

print_success "Python packages initialized"

# Create .gitignore
print_info "Creating .gitignore..."

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Airflow
logs/
airflow.db
airflow.cfg
airflow-webserver.pid
standalone_admin_password.txt

# Data
data/raw/*
data/processed/*
data/features/*
data/predictions/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/features/.gitkeep
!data/predictions/.gitkeep

# Models
models/*.pkl
models/*.h5
models/*.pb
!models/.gitkeep

# Environment
.env
.env.local
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Docker
.dockerignore

# Pytest
.pytest_cache/
.coverage
htmlcov/

# MLflow
mlruns/
mlartifacts/
EOF

print_success ".gitignore created"

# Create .gitkeep files for empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/features/.gitkeep
touch data/predictions/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep

# Create .env.example
print_info "Creating .env.example..."

cat > .env.example << 'EOF'
# OpenWeatherMap API Configuration
OPENWEATHER_API_KEY=your_api_key_here

# PostgreSQL Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=weatherflow
POSTGRES_USER=weather_user
POSTGRES_PASSWORD=change_this_password

# Airflow Configuration
AIRFLOW_UID=50000
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
AIRFLOW__CORE__FERNET_KEY=your_fernet_key_here
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True

# ML Configuration (Phase 2)
MODEL_REGISTRY_PATH=/opt/airflow/models
MLFLOW_TRACKING_URI=http://mlflow:5000

# Application Configuration
LOG_LEVEL=INFO
MAX_RETRIES=3
API_TIMEOUT=10
EOF

print_success ".env.example created"

# Create requirements.txt
print_info "Creating requirements.txt..."

cat > requirements.txt << 'EOF'
# Core Dependencies
apache-airflow==2.10.3
apache-airflow-providers-postgres==5.11.0
pandas==2.1.3
numpy==1.24.3
requests==2.31.0
python-dotenv==1.0.0

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# Data Processing
pyyaml==6.0.1

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0

# Code Quality
black==23.11.0
pylint==3.0.2
flake8==6.1.0

# Phase 2: ML Dependencies (uncomment when needed)
# tensorflow==2.15.0
# scikit-learn==1.3.2
# prophet==1.1.5
# mlflow==2.8.1

# Phase 3: Dashboard Dependencies (uncomment when needed)
# streamlit==1.28.2
# plotly==5.18.0
# folium==0.15.0
EOF

print_success "requirements.txt created"

# Create docker-compose.yaml
print_info "Creating docker-compose.yaml..."

cat > docker-compose.yaml << 'EOF'
version: '3.8'

x-airflow-common:
  &airflow-common
  image: apache/airflow:2.10.3-python3.10
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session'
    AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK: 'true'
    _PIP_ADDITIONAL_REQUIREMENTS: ''
  volumes:
    - ${AIRFLOW_PROJ_DIR:-.}/dags:/opt/airflow/dags
    - ${AIRFLOW_PROJ_DIR:-.}/logs:/opt/airflow/logs
    - ${AIRFLOW_PROJ_DIR:-.}/config:/opt/airflow/config
    - ${AIRFLOW_PROJ_DIR:-.}/plugins:/opt/airflow/plugins
    - ${AIRFLOW_PROJ_DIR:-.}/data:/opt/airflow/data
    - ${AIRFLOW_PROJ_DIR:-.}/src:/opt/airflow/src
  user: "${AIRFLOW_UID:-50000}:0"
  depends_on:
    &airflow-common-depends-on
    postgres:
      condition: service_healthy

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 10s
      retries: 5
      start_period: 5s
    restart: always
    ports:
      - "5432:5432"

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: always
    depends_on:
      <<: *airflow-common-depends-on
      airflow-init:
        condition: service_completed_successfully

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8974/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: always
    depends_on:
      <<: *airflow-common-depends-on
      airflow-init:
        condition: service_completed_successfully

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        mkdir -p /sources/logs /sources/dags /sources/plugins /sources/data
        chown -R "${AIRFLOW_UID}:0" /sources/{logs,dags,plugins,data}
        exec /entrypoint airflow version
    environment:
      <<: *airflow-common-env
      _AIRFLOW_DB_MIGRATE: 'true'
      _AIRFLOW_WWW_USER_CREATE: 'true'
      _AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow}
      _AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow}
    user: "0:0"
    volumes:
      - ${AIRFLOW_PROJ_DIR:-.}:/sources

volumes:
  postgres-db-volume:
EOF

print_success "docker-compose.yaml created"

# Create database initialization script
print_info "Creating database initialization script..."

mkdir -p scripts

cat > scripts/init-db.sql << 'EOF'
-- Create weatherflow database
CREATE DATABASE weatherflow;

-- Create user
CREATE USER weather_user WITH PASSWORD 'weather_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE weatherflow TO weather_user;

-- Connect to weatherflow database
\c weatherflow;

-- Create weather_observations table
CREATE TABLE IF NOT EXISTS weather_observations (
    id SERIAL PRIMARY KEY,
    city VARCHAR(100) NOT NULL,
    country VARCHAR(10),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    temperature DECIMAL(5, 2),
    feels_like DECIMAL(5, 2),
    humidity INTEGER,
    pressure INTEGER,
    wind_speed DECIMAL(5, 2),
    wind_direction INTEGER,
    cloudiness INTEGER,
    description VARCHAR(100),
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(city, timestamp)
);

-- Create indexes
CREATE INDEX idx_city_timestamp ON weather_observations(city, timestamp DESC);
CREATE INDEX idx_timestamp ON weather_observations(timestamp DESC);

-- Grant table privileges
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO weather_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO weather_user;
EOF

print_success "Database initialization script created"

# Create config/cities.json
print_info "Creating cities configuration..."

cat > config/cities.json << 'EOF'
{
  "cities": [
    {
      "name": "Boston",
      "country": "US",
      "priority": "high"
    },
    {
      "name": "New York",
      "country": "US",
      "priority": "high"
    },
    {
      "name": "Chicago",
      "country": "US",
      "priority": "medium"
    },
    {
      "name": "San Francisco",
      "country": "US",
      "priority": "high"
    },
    {
      "name": "Seattle",
      "country": "US",
      "priority": "medium"
    }
  ]
}
EOF

print_success "Cities configuration created"

# Create LICENSE
print_info "Creating LICENSE..."

cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Dennis Jose

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

print_success "LICENSE created"

# Create placeholder test file
print_info "Creating placeholder test files..."

cat > tests/test_sample.py << 'EOF'
"""
Sample test file - demonstrates test structure
"""
import pytest

def test_sample():
    """Sample test that always passes"""
    assert True

def test_addition():
    """Test basic arithmetic"""
    assert 1 + 1 == 2

# More tests will be added as the project develops
EOF

print_success "Test files created"

# Create .env file from .env.example
print_info "Creating .env file..."
cp .env.example .env

cat >> .env << 'EOF'

# Set UID for Airflow
AIRFLOW_UID=50000
EOF

print_success ".env file created (remember to update API key!)"

# Create initial documentation structure
print_info "Creating documentation files..."

cat > docs/api_reference.md << 'EOF'
# API Reference

This document will contain API documentation for the WeatherFlow project.

## Coming Soon
- Data fetcher API
- Validator API
- Transformer API
- Model API
EOF

cat > docs/deployment_guide.md << 'EOF'
# Deployment Guide

## Local Deployment
See README.md Quick Start section

## Production Deployment
Coming in Phase 2
EOF

cat > docs/troubleshooting.md << 'EOF'
# Troubleshooting Guide

See README.md Troubleshooting section for common issues.
EOF

print_success "Documentation structure created"

# Create a simple setup guide
print_info "Creating SETUP.md..."

cat > SETUP.md << 'EOF'
# WeatherFlow Setup Guide

## Quick Setup

1. **Get OpenWeatherMap API Key**
   - Go to https://openweathermap.org/api
   - Sign up for free account
   - Copy your API key

2. **Update .env file**
   ```bash
   # Edit .env file
   nano .env
   
   # Replace this line:
   OPENWEATHER_API_KEY=your_api_key_here
   
   # With your actual key:
   OPENWEATHER_API_KEY=abc123your_actual_key_here
   ```

3. **Start Docker**
   ```bash
   # Initialize Airflow
   docker compose up airflow-init
   
   # Start all services
   docker compose up -d
   ```

4. **Configure Airflow**
   - Open http://localhost:8080
   - Login: airflow / airflow
   - Go to Admin → Variables
   - Add: `openweather_api_key` = your_api_key

5. **Copy the DAG file**
   - Copy `weather_etl_pipeline.py` to `dags/` folder
   - Wait 30 seconds for Airflow to detect it

6. **Run the pipeline**
   - Toggle the DAG ON
   - Click the play button to trigger

7. **View results**
   ```bash
   ls -la data/
   cat data/weather_report_*.txt
   ```

## Troubleshooting

**DAG not showing up?**
```bash
docker compose logs airflow-scheduler
```

**API errors?**
- Wait 10 minutes after API key creation
- Check key is correct in Airflow Variables

**Need help?**
- Check README.md
- Check DESIGN.md
- Open GitHub issue
EOF

print_success "Setup guide created"

# Summary
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
print_success "Project structure created successfully"
echo ""
echo "Project location: $(pwd)"
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_NAME"
echo "  2. Get API key from https://openweathermap.org/api"
echo "  3. Edit .env file and add your API key"
echo "  4. Copy weather_etl_pipeline.py to dags/ folder"
echo "  5. Run: docker compose up airflow-init"
echo "  6. Run: docker compose up -d"
echo "  7. Open http://localhost:8080 (airflow/airflow)"
echo "  8. Configure API key in Airflow Variables"
echo ""
echo "Documentation:"
echo "  - README.md: Project overview and usage"
echo "  - DESIGN.md: Technical architecture"
echo "  - SETUP.md: Quick setup instructions"
echo ""
print_info "Don't forget to initialize git repository:"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial commit: WeatherFlow project setup'"
echo ""
print_success "Happy coding! 🚀"
echo ""