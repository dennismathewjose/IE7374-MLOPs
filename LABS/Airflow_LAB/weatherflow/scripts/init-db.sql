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
