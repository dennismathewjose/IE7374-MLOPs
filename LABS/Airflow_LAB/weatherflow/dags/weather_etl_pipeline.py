"""
Weather ETL Pipeline - Airflow Lab
Fetches weather data for multiple cities, processes it, and stores results
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import requests
import json
import csv
from pathlib import Path

# Default arguments for all tasks
default_args = {
    'owner': 'dennis',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

# Cities to fetch weather data for
CITIES = ['Boston', 'New York', 'Chicago', 'San Francisco', 'Seattle']

# Output directory (will be created in /opt/airflow/data inside container)
OUTPUT_DIR = Path('/opt/airflow/data')


def fetch_weather_data(**context):
    """
    Task 1: Fetch weather data from OpenWeatherMap API for multiple cities
    """
    # Get API key from Airflow Variables (we'll set this via UI)
    api_key = Variable.get("openweather_api_key")
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    weather_data = []
    
    print(f"Fetching weather data for {len(CITIES)} cities...")
    
    for city in CITIES:
        try:
            params = {
                'q': city,
                'appid': api_key,
                'units': 'metric'  # Use Celsius
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant information
            city_weather = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.now().isoformat()
            }
            
            weather_data.append(city_weather)
            print(f"✓ Fetched data for {city}: {city_weather['temperature']}°C")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to fetch data for {city}: {str(e)}")
            raise
    
    # Push data to XCom for next task
    context['ti'].xcom_push(key='raw_weather_data', value=weather_data)
    print(f"Successfully fetched weather data for {len(weather_data)} cities")
    
    return len(weather_data)


def validate_data(**context):
    """
    Task 2: Validate the fetched weather data
    """
    # Pull data from previous task
    weather_data = context['ti'].xcom_pull(key='raw_weather_data', task_ids='fetch_weather')
    
    if not weather_data:
        raise ValueError("No weather data received from fetch task")
    
    print(f"Validating {len(weather_data)} weather records...")
    
    validated_data = []
    invalid_count = 0
    
    for record in weather_data:
        # Validation rules
        is_valid = True
        issues = []
        
        # Check required fields
        required_fields = ['city', 'temperature', 'humidity', 'timestamp']
        for field in required_fields:
            if field not in record or record[field] is None:
                is_valid = False
                issues.append(f"Missing {field}")
        
        # Check temperature range (reasonable for Earth's surface)
        if 'temperature' in record:
            if record['temperature'] < -90 or record['temperature'] > 60:
                is_valid = False
                issues.append(f"Temperature out of range: {record['temperature']}°C")
        
        # Check humidity range
        if 'humidity' in record:
            if record['humidity'] < 0 or record['humidity'] > 100:
                is_valid = False
                issues.append(f"Humidity out of range: {record['humidity']}%")
        
        if is_valid:
            validated_data.append(record)
            print(f"✓ Valid: {record['city']}")
        else:
            invalid_count += 1
            print(f"✗ Invalid: {record.get('city', 'Unknown')} - {', '.join(issues)}")
    
    if invalid_count > len(weather_data) / 2:
        raise ValueError(f"Too many invalid records: {invalid_count}/{len(weather_data)}")
    
    # Push validated data to XCom
    context['ti'].xcom_push(key='validated_weather_data', value=validated_data)
    print(f"Validation complete: {len(validated_data)} valid, {invalid_count} invalid")
    
    return len(validated_data)


def transform_data(**context):
    """
    Task 3: Transform the weather data (add calculated fields, conversions)
    """
    # Pull validated data
    weather_data = context['ti'].xcom_pull(key='validated_weather_data', task_ids='validate_data')
    
    print(f"Transforming {len(weather_data)} weather records...")
    
    transformed_data = []
    
    for record in weather_data:
        # Create transformed record with additional fields
        transformed = record.copy()
        
        # Add Fahrenheit temperature
        transformed['temperature_f'] = (record['temperature'] * 9/5) + 32
        transformed['feels_like_f'] = (record['feels_like'] * 9/5) + 32
        
        # Add comfort level based on temperature
        temp_c = record['temperature']
        if temp_c < 0:
            transformed['comfort_level'] = 'Very Cold'
        elif temp_c < 10:
            transformed['comfort_level'] = 'Cold'
        elif temp_c < 20:
            transformed['comfort_level'] = 'Mild'
        elif temp_c < 30:
            transformed['comfort_level'] = 'Warm'
        else:
            transformed['comfort_level'] = 'Hot'
        
        # Add wind description
        wind_speed = record['wind_speed']
        if wind_speed < 5:
            transformed['wind_description'] = 'Calm'
        elif wind_speed < 15:
            transformed['wind_description'] = 'Moderate'
        else:
            transformed['wind_description'] = 'Strong'
        
        transformed_data.append(transformed)
        print(f"✓ Transformed: {record['city']} - {transformed['comfort_level']}")
    
    # Calculate summary statistics
    temps = [r['temperature'] for r in transformed_data]
    summary = {
        'avg_temperature': sum(temps) / len(temps),
        'max_temperature': max(temps),
        'min_temperature': min(temps),
        'total_cities': len(transformed_data),
        'timestamp': datetime.now().isoformat()
    }
    
    # Push both transformed data and summary
    context['ti'].xcom_push(key='transformed_weather_data', value=transformed_data)
    context['ti'].xcom_push(key='weather_summary', value=summary)
    
    print(f"Transformation complete. Avg temp: {summary['avg_temperature']:.1f}°C")
    
    return summary


def store_data(**context):
    """
    Task 4: Store the transformed data to CSV files
    """
    # Pull transformed data and summary
    weather_data = context['ti'].xcom_pull(key='transformed_weather_data', task_ids='transform_data')
    summary = context['ti'].xcom_pull(key='weather_summary', task_ids='transform_data')
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detail_file = OUTPUT_DIR / f'weather_data_{timestamp}.csv'
    summary_file = OUTPUT_DIR / f'weather_summary_{timestamp}.json'
    
    print(f"Storing weather data to {detail_file}...")
    
    # Write detailed data to CSV
    if weather_data:
        fieldnames = weather_data[0].keys()
        with open(detail_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(weather_data)
        
        print(f"✓ Stored {len(weather_data)} records to {detail_file}")
    
    # Write summary to JSON
    with open(summary_file, 'w') as jsonfile:
        json.dump(summary, jsonfile, indent=2)
    
    print(f"✓ Stored summary to {summary_file}")
    
    return {
        'detail_file': str(detail_file),
        'summary_file': str(summary_file),
        'records_stored': len(weather_data)
    }


def generate_report(**context):
    """
    Task 5: Generate a simple text report
    """
    # Pull summary data
    summary = context['ti'].xcom_pull(key='weather_summary', task_ids='transform_data')
    weather_data = context['ti'].xcom_pull(key='transformed_weather_data', task_ids='transform_data')
    
    report_lines = [
        "=" * 60,
        "WEATHER REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Cities Analyzed: {summary['total_cities']}",
        "",
        "TEMPERATURE SUMMARY:",
        f"  Average: {summary['avg_temperature']:.1f}°C ({summary['avg_temperature']*9/5+32:.1f}°F)",
        f"  Maximum: {summary['max_temperature']:.1f}°C",
        f"  Minimum: {summary['min_temperature']:.1f}°C",
        "",
        "CITY DETAILS:",
    ]
    
    # Add details for each city
    for record in sorted(weather_data, key=lambda x: x['temperature'], reverse=True):
        report_lines.append(
            f"  {record['city']:15} {record['temperature']:6.1f}°C  "
            f"{record['description']:20} {record['comfort_level']}"
        )
    
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    print(report)
    
    # Save report to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = OUTPUT_DIR / f'weather_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to {report_file}")
    
    return str(report_file)


# Define the DAG
with DAG(
    dag_id='weather_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline to fetch, process, and store weather data',
    schedule_interval='0 */6 * * *',  # Run every 6 hours
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=['weather', 'etl', 'lab'],
) as dag:
    
    # Task 1: Fetch weather data from API
    fetch_weather = PythonOperator(
        task_id='fetch_weather',
        python_callable=fetch_weather_data,
        provide_context=True,
    )
    
    # Task 2: Validate the fetched data
    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )
    
    # Task 3: Transform the data
    transform_data_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        provide_context=True,
    )
    
    # Task 4: Store data to CSV
    store_data_task = PythonOperator(
        task_id='store_data',
        python_callable=store_data,
        provide_context=True,
    )
    
    # Task 5: Generate report
    generate_report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
        provide_context=True,
    )
    
    # Define task dependencies
    fetch_weather >> validate_data_task >> transform_data_task >> store_data_task >> generate_report_task