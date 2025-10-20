# ML Model Monitoring with ELK Stack

A comprehensive real-time monitoring system for Machine Learning model inference metrics using Elasticsearch, Logstash, and Kibana (ELK Stack).

##  Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Visualizations](#visualizations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

##  Overview

This project implements a production-ready monitoring system for ML models that tracks:
- **Model Performance**: Prediction confidence, accuracy, and drift
- **Inference Latency**: Response times and performance bottlenecks
- **Resource Utilization**: CPU, memory, and GPU usage
- **Error Tracking**: Failed predictions and system errors
- **Model Distribution**: Request patterns across different models

### Key Features
- Real-time monitoring of multiple ML models
- Automated log processing and enrichment
- Interactive dashboards with drill-down capabilities
- Scalable architecture using Docker containers
- Support for multiple environments (production, staging, development)

##  Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   ML Models     │────▶│   Logstash   │────▶│ Elasticsearch │────▶│    Kibana    │
│  (Log Generator) │ TCP │  (Pipeline)  │     │   (Storage)   │     │ (Dashboard)  │
└─────────────────┘ 5001└──────────────┘     └───────────────┘     └──────────────┘
                              ▲                                              ▲
                              │                                              │
                         Processing &                                   Visualization
                         Enrichment                                    & Analytics
```

## Project Structure

```
ml-elk-monitoring/
├── docker-compose.yml              # Docker services configuration
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                     # Git ignore patterns
├── setup_env.sh                   # Environment setup script
│
├── elasticsearch/
│   ├── config/
│   │   └── elasticsearch.yml     # Elasticsearch configuration
│   └── templates/
│       ├── ml-metrics-template.json      # Index template for ML metrics
│       ├── ml-training-template.json     # Index template for training metrics
│       └── data-quality-template.json    # Index template for data quality
│
├── logstash/
│   ├── config/
│   │   └── logstash.yml          # Logstash configuration
│   └── pipeline/
│       └── ml-pipeline.conf      # ML log processing pipeline
│
├── kibana/
│   └── config/
│       └── kibana.yml            # Kibana configuration
│
├── ml-logs/                      # Directory for log files
│   └── *.json                    # Generated ML logs
│
├── scripts/
│   ├── ml_log_generator.py       # Enhanced ML log generator
│   ├── simple_ml_generator.py    # Simple log generator (no dependencies)
│   ├── insert_data.py            # Direct Elasticsearch data inserter
│   ├── manage-templates.sh       # Template management script
│   ├── monitor_logs.sh           # Real-time monitoring script
│   ├── verify-elasticsearch.sh   # Elasticsearch health check
│   ├── verify-logstash.sh        # Logstash verification
│   ├── test_ml_pipeline.sh       # Complete pipeline test
│   └── setup_kibana.sh           # Kibana setup automation
│
├── test-data/                    # Sample test data
│   └── sample-logs.json          # Example log entries
│
└── venv/                         # Python virtual environment
    └── ...                       # Virtual environment files
```

## Prerequisites

- **Docker Desktop** (includes Docker and Docker Compose)
  - macOS: [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
  - Linux: Docker Engine + Docker Compose
  - Windows: Docker Desktop with WSL2
- **Python 3.8+** 
- **4GB+ RAM** available for Docker containers
- **10GB+ disk space** for data and containers
- **Terminal** with bash/zsh shell

### System Requirements
- CPU: 2+ cores recommended
- Memory: 8GB total (4GB for ELK Stack)
- Network: Ports 9200, 5601, 5001, 5002, 9600 available

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ml-elk-monitoring
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start ELK Stack
```bash
# Start all services
docker-compose up -d

# Wait for services to initialize (60 seconds)
sleep 60

# Verify all services are running
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### 4. Apply Elasticsearch Templates
```bash
# Apply ML metrics template
curl -X PUT "localhost:9200/_index_template/ml-metrics-template" \
  -H 'Content-Type: application/json' \
  -d @elasticsearch/templates/ml-metrics-template.json

# Verify template
./scripts/manage-templates.sh verify ml-metrics-template
```

### 5. Generate Initial Data
```bash
# Generate 500 sample ML logs
python scripts/ml_log_generator.py --batch 500

# Or use simple generator if dependencies fail
python scripts/simple_ml_generator.py --count 500
```

### 6. Access Kibana
Open your browser and navigate to: http://localhost:5601

## Usage

### Generating ML Logs

#### Batch Generation
```bash
# Generate specific number of logs
python scripts/ml_log_generator.py --batch 1000

# Generate with verbose output
python scripts/ml_log_generator.py --batch 100 --verbose
```

#### Continuous Generation
```bash
# Generate logs continuously at specified rate
python scripts/ml_log_generator.py --continuous --rate 20

# Generate for specific duration (seconds)
python scripts/ml_log_generator.py --continuous --rate 10 --duration 60
```

#### Scenario Testing
```bash
# Test high load scenario
python scripts/ml_log_generator.py --scenario high_load

# Test degradation scenario
python scripts/ml_log_generator.py --scenario degradation
```

### Monitoring

#### Real-time Monitoring
```bash
# Monitor logs in terminal
./scripts/monitor_logs.sh

# Check pipeline health
./scripts/test_ml_pipeline.sh
```

#### System Verification
```bash
# Verify Elasticsearch
./scripts/verify-elasticsearch.sh

# Verify complete stack
./scripts/verify_all_services.sh
```

## Data Pipeline

### Log Flow
1. **Generation**: ML models generate JSON logs with inference metrics
2. **Transport**: Logs sent via TCP to Logstash (port 5001)
3. **Processing**: Logstash enriches logs with:
   - Confidence level categorization (high/medium/low)
   - Latency categorization (fast/normal/slow/very_slow)
   - Prediction accuracy calculation
   - Error type classification
4. **Storage**: Processed logs stored in Elasticsearch with daily indices
5. **Visualization**: Kibana queries Elasticsearch for real-time dashboards

### Log Structure
```json
{
  "@timestamp": "2024-01-15T10:00:00Z",
  "model_name": "fraud_detector",
  "model_version": "2.3.1",
  "type": "model_inference",
  "prediction_confidence": 0.95,
  "prediction_class": "legitimate",
  "true_label": "legitimate",
  "inference_time_ms": 23.5,
  "cpu_usage": 45.2,
  "memory_usage_mb": 1024,
  "gpu_usage": 0,
  "environment": "production",
  "region": "us-east-1",
  "request_id": "req_abc123",
  "user_id": "user_456",
  "data_quality_score": 0.92,
  "error_message": null,
  "error": false
}
```

### Index Patterns
- `ml-metrics-*`: Main metrics data
- `ml-training-*`: Training metrics
- `data-quality-*`: Data quality monitoring

## Visualizations

### Dashboard Components

#### 1. Model Performance Gauge
- **Type**: Horizontal Gauge
- **Metric**: Average prediction confidence
- **Ranges**: 
  - 0-0.7: Red (Poor)
  - 0.7-0.85: Yellow (Fair)
  - 0.85-1.0: Green (Good)

#### 2. Inference Latency Trend
- **Type**: Line Chart
- **Metrics**: Average and P95 latency
- **X-axis**: Time
- **Y-axis**: Latency (ms)

#### 3. Model Distribution
- **Type**: Pie Chart
- **Shows**: Request distribution across models
- **Aggregation**: Count by model_name

#### 4. Resource Utilization
- **Type**: Metric
- **Displays**: Average CPU%, Memory usage

#### 5. Performance Table
- **Type**: Data Table
- **Columns**: Model, Environment, Count, Avg Latency, Avg Confidence
- **Sorting**: By request count

### Creating Custom Visualizations
1. Navigate to Kibana → Dashboard
2. Click "Create visualization"
3. Select visualization type
4. Configure metrics and buckets
5. Save and add to dashboard

## 🔧 Configuration

### Environment Variables
```bash
# Elasticsearch
ES_JAVA_OPTS=-Xms1g -Xmx1g

# Logstash
LS_JAVA_OPTS=-Xms512m -Xmx512m

# Ports
ELASTICSEARCH_PORT=9200
KIBANA_PORT=5601
LOGSTASH_TCP_PORT=5001
```

### Scaling Configuration
For production environments, adjust in `docker-compose.yml`:
```yaml
elasticsearch:
  environment:
    - "ES_JAVA_OPTS=-Xms4g -Xmx4g"
  resources:
    limits:
      memory: 8g
```

## Troubleshooting

### Common Issues

#### No Data in Kibana
```bash
# Check Elasticsearch has data
curl -s "localhost:9200/ml-metrics-*/_count" | python3 -m json.tool

# Check Logstash is running
docker-compose logs --tail=50 logstash

# Restart services
docker-compose restart
```

#### Logstash Connection Issues
```bash
# Test port connectivity
nc -zv localhost 5001

# Check Logstash pipeline
curl -s "localhost:9600/_node/pipelines?pretty"
```

#### Memory Issues
```bash
# Reduce heap sizes in docker-compose.yml
ES_JAVA_OPTS=-Xms512m -Xmx512m
LS_JAVA_OPTS=-Xms256m -Xmx256m
```

#### Port Conflicts
```bash
# Check port usage
lsof -i :9200
lsof -i :5601
lsof -i :5001

# Change ports in docker-compose.yml if needed
```

### Debugging Commands
```bash
# View all indices
curl -s "localhost:9200/_cat/indices?v"

# Check cluster health
curl -s "localhost:9200/_cluster/health?pretty"

# View Logstash logs
docker-compose logs -f logstash

# Test data insertion
echo '{"timestamp":"2024-01-15T10:00:00Z","model_name":"test","inference_time_ms":25}' | nc localhost 5001
```

### Adding New Models
1. Update `ml_log_generator.py` with new model configuration
2. Add model-specific fields to Elasticsearch template
3. Create model-specific visualizations in Kibana


### Production Deployment
1. Enable security features in Elasticsearch
2. Set up SSL/TLS for all services
3. Configure persistent volumes for data
4. Implement backup strategies
5. Set up monitoring for the ELK stack itself

---

**Last Updated**: October 2025
**Version**: 1.0.0
**Status**: Production Ready

