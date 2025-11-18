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
