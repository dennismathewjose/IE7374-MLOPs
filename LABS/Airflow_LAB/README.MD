# WeatherFlow: Real-Time Weather ETL & Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apache Airflow](https://img.shields.io/badge/airflow-2.10.3-red.svg)](https://airflow.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade weather data pipeline that combines Apache Airflow orchestration with machine learning to fetch real-time weather data, process it, and predict future weather conditions for the next 3-7 days.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development Phases](#development-phases)
- [Testing](#testing)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

##  Overview

**WeatherFlow** is an end-to-end weather data engineering and machine learning project designed to:

1. **Ingest** real-time weather data from OpenWeatherMap API
2. **Process** and validate data through automated ETL pipelines
3. **Store** historical weather data in PostgreSQL
4. **Predict** future weather conditions using LSTM/Prophet models
5. **Visualize** current and predicted weather through interactive dashboards

**Built With:**
- Apache Airflow 2.10.3 (orchestration)
- Docker & Docker Compose (containerization)
- PostgreSQL 15 (data storage)
- Python 3.10+ (core language)
- TensorFlow/Keras (ML models - Phase 2)
- Streamlit (dashboard - Phase 3)

**Use Cases:**
- Learning Apache Airflow and data pipeline orchestration
- Understanding ETL best practices
- Building production ML pipelines
- Time series forecasting with weather data
- Academic course project/lab assignment

---

##  Features

### Phase 1: ETL Pipeline (Current)
- ✅ Automated weather data fetching from OpenWeatherMap API
- ✅ Multi-city data collection (5+ cities)
- ✅ Data validation and quality checks
- ✅ Feature engineering and transformations
- ✅ CSV and database storage
- ✅ Automated report generation
- ✅ Error handling and retry logic
- ✅ Comprehensive logging

### Phase 2: ML Pipeline (Planned)
- 🔄 Historical data analysis
- 🔄 LSTM model for temperature prediction
- 🔄 Prophet model for time series forecasting
- 🔄 Model training and evaluation pipeline
- 🔄 Real-time prediction generation
- 🔄 Model versioning and registry

### Phase 3: Dashboard (Planned)
- 🔄 Interactive Streamlit dashboard
- 🔄 Real-time weather visualization
- 🔄 7-day forecast display
- 🔄 Historical trend charts
- 🔄 Alert notifications

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WeatherFlow Architecture                  │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                            │
         ┌──────▼──────┐            ┌───────▼────────┐
         │ Data Source │            │   Orchestrator │
         │  OpenWM API │───────────▶│  Airflow DAGs  │
         └─────────────┘            └───────┬────────┘
                                            │
                        ┌───────────────────┼───────────────────┐
                        │                   │                   │
                   ┌────▼─────┐      ┌─────▼──────┐     ┌─────▼─────┐
                   │ Validate │      │ Transform  │     │   Store   │
                   │   Data   │─────▶│    Data    │────▶│  CSV/DB   │
                   └──────────┘      └────────────┘     └─────┬─────┘
                                                               │
                        ┌──────────────────────────────────────┤
                        │                                      │
                   ┌────▼─────┐                         ┌─────▼─────┐
                   │ ML Model │                         │ Dashboard │
                   │ Training │                         │ Streamlit │
                   └──────────┘                         └───────────┘
```

**Key Components:**
1. **Data Ingestion**: OpenWeatherMap API integration
2. **ETL Pipeline**: Airflow DAGs for orchestration
3. **Storage Layer**: PostgreSQL + CSV files
4. **ML Pipeline**: Model training and inference
5. **Visualization**: Interactive dashboards

---

##  Quick Start

### Prerequisites

- Docker Desktop (20.10+)
- Docker Compose (2.0+)
- Git
- OpenWeatherMap API key (free tier)

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/weatherflow.git
cd weatherflow
```

**2. Create environment file:**
```bash
cp .env.example .env
# Edit .env and add your OpenWeatherMap API key
```

**3. Start Docker containers:**
```bash
# Initialize Airflow database
docker compose up airflow-init

# Start all services
docker compose up -d
```

**4. Access Airflow UI:**
- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

**5. Configure API key in Airflow:**
- Go to Admin → Variables
- Add variable: `openweather_api_key` = `your_api_key_here`

**6. Trigger the pipeline:**
- Navigate to DAGs
- Find `weather_etl_pipeline`
- Toggle it ON
- Click ▶ to trigger manually

**7. View results:**
```bash
# Check output files
ls -la ./data/

# View report
cat ./data/weather_report_*.txt

# Check database
docker compose exec postgres psql -U weather_user -d weatherflow
```

---

##  Project Structure

```
weatherflow/
├── README.md                    # This file
├── DESIGN.md                    # Low-level design document
├── docker-compose.yaml          # Docker services configuration
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
│
├── dags/                        # Airflow DAG definitions
│   └── weather_etl_pipeline.py  # Main ETL pipeline
│
├── src/                         # Source code (Phase 2)
│   ├── data/                    # Data processing modules
│   ├── models/                  # ML models
│   ├── storage/                 # Database operations
│   └── utils/                   # Utility functions
│
├── tests/                       # Unit tests
│   └── test_*.py               # Test files
│
├── data/                        # Data storage (gitignored)
│   ├── raw/                    # Raw API responses
│   ├── processed/              # Cleaned data
│   └── predictions/            # Model outputs
│
├── models/                      # Trained models (Phase 2)
├── logs/                        # Application logs
├── config/                      # Configuration files
├── notebooks/                   # Jupyter notebooks
├── dashboard/                   # Streamlit app (Phase 3)
├── scripts/                     # Utility scripts
└── docs/                        # Documentation
```

---

##  Configuration

### Environment Variables (.env)

```bash
# OpenWeatherMap API
OPENWEATHER_API_KEY=your_api_key_here

# PostgreSQL Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=weatherflow
POSTGRES_USER=weather_user
POSTGRES_PASSWORD=secure_password

# Airflow
AIRFLOW_UID=50000
AIRFLOW__CORE__EXECUTOR=LocalExecutor
```

### Airflow Variables

Set these in Airflow UI (Admin → Variables):

| Variable | Value | Description |
|----------|-------|-------------|
| `openweather_api_key` | Your API key | Required for API access |
| `cities_list` | `["Boston","NYC",...]` | Cities to monitor |
| `alert_threshold_temp` | `35.0` | Temperature alert threshold |

### Cities Configuration

Edit `config/cities.json`:
```json
{
  "cities": [
    {"name": "Boston", "country": "US", "priority": "high"},
    {"name": "New York", "country": "US", "priority": "high"}
  ]
}
```

---

##  Usage

### Running the ETL Pipeline

**Manual Trigger:**
```bash
# Via UI: Click "Trigger DAG" in Airflow UI

# Via CLI:
docker compose exec airflow-webserver airflow dags trigger weather_etl_pipeline
```

**Scheduled Execution:**
- Default: Every 6 hours (`0 */6 * * *`)
- Modify in `dags/weather_etl_pipeline.py`:
```python
schedule_interval='@daily'  # Run daily at midnight
```

### Viewing Results

**Check CSV outputs:**
```bash
ls -la ./data/
cat ./data/weather_data_20251117_143000.csv
cat ./data/weather_report_20251117_143000.txt
```

**Query database:**
```bash
docker compose exec postgres psql -U weather_user -d weatherflow

-- View latest observations
SELECT city, temperature, humidity, timestamp 
FROM weather_observations 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Monitoring Pipeline

**Airflow UI:**
- Graph View: Task dependencies
- Grid View: Historical runs
- Task Logs: Detailed execution logs

**Check logs:**
```bash
# Airflow scheduler logs
docker compose logs airflow-scheduler

# Task-specific logs
docker compose exec airflow-scheduler cat /opt/airflow/logs/...
```

---

##  Development Phases

###  Phase 1: ETL Pipeline (Current - Weeks 1-2)

**Goals:**
- Set up Airflow with Docker
- Implement 5-task ETL pipeline
- Store data in CSV and PostgreSQL
- Generate automated reports

**Deliverables:**
- Working ETL pipeline
- Data quality validation
- Error handling and logging
- Project documentation

**Status:**  Complete

---

###  Phase 2: ML Pipeline (Weeks 3-4)

**Goals:**
- Collect historical data (30+ days)
- Train LSTM temperature prediction model
- Train Prophet time series model
- Implement prediction pipeline
- Store predictions in database

**Tasks:**
1. Create `weather_ml_pipeline.py` DAG
2. Implement feature engineering
3. Build and train LSTM model
4. Build and train Prophet model
5. Create prediction inference DAG
6. Set up model versioning

**Status:** 🔄 Planned

---

###  Phase 3: Dashboard (Weeks 5-6)

**Goals:**
- Build interactive Streamlit dashboard
- Visualize current weather
- Display 7-day predictions
- Show historical trends
- Implement alert system

**Features:**
- Real-time weather map
- Temperature trend charts
- Prediction confidence scores
- Email/Slack alerts

**Status:** 🔄 Planned

---

##  Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_validator.py
```

### Test Structure

```
tests/
├── test_fetcher.py      # API fetching tests
├── test_validator.py    # Data validation tests
├── test_transformer.py  # Data transformation tests
└── test_pipeline.py     # End-to-end tests
```

### Writing Tests

```python
# tests/test_validator.py
import pytest
from src.data.validator import validate_temperature

def test_valid_temperature():
    assert validate_temperature(25.0) == True

def test_invalid_temperature():
    assert validate_temperature(150.0) == False
```

---

## 🚢 Deployment

### Local Development
```bash
# Already running if you followed Quick Start
docker compose up -d
```

### Production Deployment (GCP)

**Option 1: Cloud Composer (Managed Airflow)**
```bash
# Create Composer environment
gcloud composer environments create weatherflow \
    --location us-central1 \
    --python-version 3.10

# Upload DAGs
gcloud composer environments storage dags import \
    --environment weatherflow \
    --location us-central1 \
    --source dags/
```

**Option 2: GKE (Kubernetes)**
```bash
# Deploy using Helm
helm install airflow apache-airflow/airflow -f values.yaml

# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml

# Deploy application
kubectl apply -f k8s/weatherflow.yaml
```

**Cost Estimate (Production):**
- Cloud Composer: ~$150/month
- Cloud SQL: ~$10/month
- Storage: ~$0.30/month
- **Total: ~$160-220/month**

---

##  Monitoring

### Metrics Tracked

**Pipeline Health:**
- DAG success rate: Target >95%
- Task execution time: Target <2 minutes
- Data freshness: Alert if >6 hours old

**Data Quality:**
- Validation failure rate: Alert if >15%
- API success rate: Alert if <90%
- Missing data percentage: Alert if >10%

**System Performance:**
- CPU usage: Alert if >80%
- Memory usage: Alert if >85%
- Disk usage: Alert if >80%

### Alerting

**Critical Alerts:**
- Pipeline failure (2+ consecutive runs)
- Database connection lost
- API rate limit exceeded

**Warning Alerts:**
- High validation failure rate
- Slow API response (>5s)
- Extreme weather detected

**Notification Channels:**
- Email (built-in Airflow alerts)
- Slack (webhook integration)
- Dashboard (Grafana)

---

##  Contributing

We welcome contributions! Here's how:

**1. Fork the repository**

**2. Create a feature branch:**
```bash
git checkout -b feature/amazing-feature
```

**3. Make your changes:**
- Add tests for new features
- Update documentation
- Follow PEP 8 style guide

**4. Run tests:**
```bash
pytest tests/
black src/
pylint src/
```

**5. Commit and push:**
```bash
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

**6. Open a Pull Request**

### Code Style

- Python: PEP 8
- Docstrings: Google style
- Type hints: Required for functions
- Max line length: 100 characters

---

##  Troubleshooting

### Common Issues

**Issue: DAG not appearing in Airflow UI**
```bash
# Check for syntax errors
docker compose exec airflow-webserver python /opt/airflow/dags/weather_etl_pipeline.py

# Restart scheduler
docker compose restart airflow-scheduler
```

**Issue: API key not working**
```bash
# Verify variable is set
docker compose exec airflow-webserver airflow variables get openweather_api_key

# Test API manually
curl "http://api.openweathermap.org/data/2.5/weather?q=Boston&appid=YOUR_KEY"
```

**Issue: Database connection failed**
```bash
# Check PostgreSQL is running
docker compose ps postgres

# Test connection
docker compose exec postgres psql -U weather_user -d weatherflow -c "SELECT 1;"
```

**Issue: No data in output files**
```bash
# Check Airflow logs
docker compose logs airflow-scheduler | grep ERROR

# Check task logs in UI
# Airflow UI → DAGs → weather_etl_pipeline → Graph → Click task → Logs
```

**Issue: Docker out of disk space**
```bash
# Clean up old containers and images
docker system prune -a

# Remove old logs
rm -rf logs/airflow/*
```

### Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: weatherflow-support@example.com

---

##  Additional Resources

**Learning Materials:**
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [OpenWeatherMap API Guide](https://openweathermap.org/guide)
- [Docker Compose Tutorial](https://docs.docker.com/compose/)

**Related Projects:**
- [awesome-airflow](https://github.com/jghoman/awesome-apache-airflow)
- [weather-pipeline](https://github.com/search?q=weather+pipeline)

**Academic Papers:**
- "Deep Learning for Weather Forecasting" (2023)
- "Time Series Prediction with LSTM" (2022)

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Authors

- **Dennis Jose** -

---

## 📧 Contact

**Developer:** Dennis Jose  
**Email:** jose.d@northeastern.edu  
**Project Link:** https://github.com/yourusername/weatherflow

---

**Last Updated:** November 17, 2025  
**Project Status:** Phase 1 Complete, Phase 2 In Planning