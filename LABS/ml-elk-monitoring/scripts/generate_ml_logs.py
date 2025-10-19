#!/usr/bin/env python3
"""
ML Log Generator for ELK Stack Lab
Generates realistic ML model monitoring logs
"""

import json
import random
import socket
import time
from datetime import datetime, timedelta
import numpy as np

class MLLogGenerator:
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        
        # Model configurations
        self.models = [
            {
                "name": "fraud_detector",
                "version": "2.3.1",
                "type": "classification",
                "classes": ["legitimate", "fraudulent"],
                "baseline_latency": 25,
                "error_rate": 0.02
            },
            {
                "name": "price_predictor",
                "version": "1.5.0",
                "type": "regression",
                "baseline_latency": 35,
                "error_rate": 0.03
            },
            {
                "name": "image_classifier",
                "version": "3.0.2",
                "type": "classification",
                "classes": ["cat", "dog", "bird", "fish", "other"],
                "baseline_latency": 45,
                "error_rate": 0.05
            },
            {
                "name": "sentiment_analyzer",
                "version": "1.2.0",
                "type": "classification",
                "classes": ["positive", "negative", "neutral"],
                "baseline_latency": 15,
                "error_rate": 0.01
            },
            {
                "name": "recommendation_engine",
                "version": "2.1.0",
                "type": "ranking",
                "baseline_latency": 55,
                "error_rate": 0.04
            }
        ]
        
        self.environments = ["production", "staging", "development"]
        self.regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
        self.servers = ["ml-server-01", "ml-server-02", "ml-server-03", "ml-server-04"]
        
    def generate_inference_log(self, model_config):
        """Generate a single inference log entry"""
        
        # Timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Model details
        model_name = model_config["name"]
        model_version = model_config["version"]
        
        # Simulate different scenarios
        scenario = random.random()
        
        if scenario < 0.8:  # Normal operation (80%)
            # Good predictions
            if model_config["type"] == "classification":
                prediction_class = random.choice(model_config["classes"])
                true_label = prediction_class if random.random() < 0.9 else random.choice(model_config["classes"])
                prediction_confidence = random.uniform(0.85, 0.99)
            else:
                prediction_class = None
                true_label = None
                prediction_confidence = random.uniform(0.80, 0.95)
            
            inference_time = np.random.normal(model_config["baseline_latency"], 5)
            inference_time = max(5, inference_time)  # Minimum 5ms
            
            cpu_usage = random.uniform(30, 70)
            memory_usage = random.uniform(500, 2000)
            gpu_usage = random.uniform(20, 60) if "image" in model_name else 0
            
            error_message = None
            
        elif scenario < 0.95:  # Degraded performance (15%)
            # Slower predictions, lower confidence
            if model_config["type"] == "classification":
                prediction_class = random.choice(model_config["classes"])
                true_label = random.choice(model_config["classes"])
                prediction_confidence = random.uniform(0.50, 0.75)
            else:
                prediction_class = None
                true_label = None
                prediction_confidence = random.uniform(0.45, 0.70)
            
            inference_time = random.uniform(100, 500)
            cpu_usage = random.uniform(70, 90)
            memory_usage = random.uniform(2000, 4000)
            gpu_usage = random.uniform(60, 85) if "image" in model_name else 0
            error_message = None
            
        else:  # Errors (5%)
            prediction_class = None
            true_label = None
            prediction_confidence = 0
            inference_time = random.uniform(500, 5000)
            cpu_usage = random.uniform(85, 100)
            memory_usage = random.uniform(4000, 8000)
            gpu_usage = random.uniform(85, 100) if "image" in model_name else 0
            
            error_messages = [
                "Connection timeout to feature store",
                "Model loading failed: Out of memory",
                "Preprocessing pipeline error",
                "Invalid input format",
                "GPU memory allocation failed",
                "Feature extraction timeout"
            ]
            error_message = random.choice(error_messages)
        
        # Build log entry
        log_entry = {
            "timestamp": timestamp,
            "model_name": model_name,
            "model_version": model_version,
            "type": "model_inference",
            "prediction_confidence": round(prediction_confidence, 4),
            "inference_time_ms": round(inference_time, 2),
            "cpu_usage": round(cpu_usage, 2),
            "memory_usage_mb": round(memory_usage, 2),
            "gpu_usage": round(gpu_usage, 2),
            "environment": random.choice(self.environments),
            "region": random.choice(self.regions),
            "server": random.choice(self.servers),
            "request_id": f"req_{random.randint(100000, 999999)}",
            "user_id": f"user_{random.randint(1, 1000)}",
            "data_quality_score": round(random.uniform(0.7, 1.0), 3)
        }
        
        # Add classification-specific fields
        if model_config["type"] == "classification" and prediction_class:
            log_entry["prediction_class"] = prediction_class
            log_entry["true_label"] = true_label
        
        # Add error if present
        if error_message:
            log_entry["error_message"] = error_message
            log_entry["error"] = True
        
        # Add feature values for some logs
        if random.random() < 0.3:
            log_entry["feature_values"] = {
                "feature_1": round(random.uniform(-1, 1), 4),
                "feature_2": round(random.uniform(0, 100), 2),
                "feature_3": random.randint(0, 10),
                "feature_4": round(random.uniform(0, 1), 4)
            }
        
        return log_entry
    
    def send_log(self, log_entry):
        """Send log to Logstash via TCP"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            message = json.dumps(log_entry) + '\n'
            sock.send(message.encode('utf-8'))
            sock.close()
            return True
        except Exception as e:
            print(f"Error sending log: {e}")
            return False
    
    def generate_batch(self, count=100, delay=0.1):
        """Generate and send a batch of logs"""
        print(f"Generating {count} ML inference logs...")
        success_count = 0
        
        for i in range(count):
            # Select random model
            model = random.choice(self.models)
            
            # Generate log
            log_entry = self.generate_inference_log(model)
            
            # Send to Logstash
            if self.send_log(log_entry):
                success_count += 1
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Sent {i + 1}/{count} logs...")
                    
                # Print sample log every 20 entries
                if (i + 1) % 20 == 0:
                    print(f"Sample log: {log_entry['model_name']} - "
                          f"Confidence: {log_entry['prediction_confidence']:.2f} - "
                          f"Latency: {log_entry['inference_time_ms']:.1f}ms")
            
            time.sleep(delay)
        
        print(f"Successfully sent {success_count}/{count} logs")
        return success_count
    
    def generate_continuous(self, rate=10):
        """Generate continuous stream of logs"""
        print(f"Generating continuous ML logs at {rate} logs/second...")
        print("Press Ctrl+C to stop")
        
        interval = 1.0 / rate
        count = 0
        
        try:
            while True:
                model = random.choice(self.models)
                log_entry = self.generate_inference_log(model)
                
                if self.send_log(log_entry):
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"Sent {count} logs...")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\nStopped. Total logs sent: {count}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ML monitoring logs')
    parser.add_argument('--host', default='localhost', help='Logstash host')
    parser.add_argument('--port', type=int, default=5001, help='Logstash port')
    parser.add_argument('--batch', type=int, help='Generate batch of N logs')
    parser.add_argument('--continuous', action='store_true', help='Generate continuous logs')
    parser.add_argument('--rate', type=int, default=10, help='Logs per second for continuous mode')
    
    args = parser.parse_args()
    
    generator = MLLogGenerator(args.host, args.port)
    
    if args.batch:
        generator.generate_batch(args.batch)
    elif args.continuous:
        generator.generate_continuous(args.rate)
    else:
        # Default: generate 500 logs
        generator.generate_batch(500, delay=0.05)

if __name__ == "__main__":
    main()
