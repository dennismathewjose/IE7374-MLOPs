#!/usr/bin/env python3
"""
Enhanced ML Log Generator for ELK Stack Lab
Generates realistic ML model monitoring logs with advanced patterns
"""

import json
import random
import socket
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm
import psutil
import argparse
from colorama import init, Fore, Style
from tabulate import tabulate

# Initialize colorama for colored output
init(autoreset=True)

class EnhancedMLLogGenerator:
    def __init__(self, host='localhost', port=5001, verbose=False):
        self.host = host
        self.port = port
        self.verbose = verbose
        self.fake = Faker()
        
        # Statistics tracking
        self.stats = {
            'total_sent': 0,
            'errors': 0,
            'models': {}
        }
        
        # Enhanced model configurations
        self.models = [
            {
                "name": "fraud_detector",
                "version": "2.3.1",
                "type": "classification",
                "classes": ["legitimate", "fraudulent"],
                "baseline_latency": 25,
                "latency_std": 5,
                "accuracy": 0.94,
                "error_rate": 0.02,
                "drift_rate": 0.001
            },
            {
                "name": "price_predictor",
                "version": "1.5.0",
                "type": "regression",
                "baseline_latency": 35,
                "latency_std": 8,
                "mse": 125.5,
                "mae": 8.3,
                "error_rate": 0.03,
                "drift_rate": 0.002
            },
            {
                "name": "image_classifier",
                "version": "3.0.2",
                "type": "classification",
                "classes": ["cat", "dog", "bird", "fish", "other"],
                "baseline_latency": 45,
                "latency_std": 10,
                "accuracy": 0.91,
                "error_rate": 0.05,
                "drift_rate": 0.003,
                "uses_gpu": True
            },
            {
                "name": "sentiment_analyzer",
                "version": "1.2.0",
                "type": "classification",
                "classes": ["positive", "negative", "neutral"],
                "baseline_latency": 15,
                "latency_std": 3,
                "accuracy": 0.88,
                "error_rate": 0.01,
                "drift_rate": 0.001
            },
            {
                "name": "recommendation_engine",
                "version": "2.1.0",
                "type": "ranking",
                "baseline_latency": 55,
                "latency_std": 12,
                "mrr": 0.75,  # Mean Reciprocal Rank
                "error_rate": 0.04,
                "drift_rate": 0.002
            },
            {
                "name": "anomaly_detector",
                "version": "1.8.3",
                "type": "anomaly_detection",
                "baseline_latency": 30,
                "latency_std": 7,
                "precision": 0.92,
                "recall": 0.87,
                "error_rate": 0.03,
                "drift_rate": 0.002
            }
        ]
        
        self.environments = ["production", "staging", "development", "qa"]
        self.regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "eu-central-1"]
        self.servers = [f"ml-server-{i:02d}" for i in range(1, 11)]
        
        # Realistic error messages
        self.error_messages = [
            "Connection timeout to feature store",
            "Model loading failed: Out of memory",
            "Preprocessing pipeline error: Invalid feature dimensions",
            "GPU memory allocation failed",
            "Feature extraction timeout exceeded 30s",
            "Redis cache connection refused",
            "Kafka consumer lag detected",
            "Model artifact corrupted",
            "TensorFlow serving unavailable",
            "PyTorch CUDA error: out of memory",
            "Data validation failed: schema mismatch",
            "Feature drift detected beyond threshold"
        ]
        
    def generate_inference_log(self, model_config, timestamp=None):
        """Generate a single inference log entry with realistic patterns"""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        timestamp_str = timestamp.isoformat() + "Z"
        
        # Model details
        model_name = model_config["name"]
        model_version = model_config["version"]
        
        # Time-based patterns (simulate peak hours)
        hour = timestamp.hour
        is_peak_hour = hour in [9, 10, 11, 14, 15, 16]  # Business hours
        traffic_multiplier = 1.5 if is_peak_hour else 1.0
        
        # Simulate model performance degradation over time
        days_since_deployment = random.randint(0, 90)
        performance_degradation = 1 + (days_since_deployment * model_config.get("drift_rate", 0.001))
        
        # Determine scenario
        scenario = random.random()
        
        if scenario < 0.75:  # Normal operation (75%)
            # Generate realistic inference metrics
            if model_config["type"] == "classification":
                classes = model_config.get("classes", ["class_a", "class_b"])
                
                # Simulate realistic class distribution
                if random.random() < model_config.get("accuracy", 0.9):
                    # Correct prediction
                    prediction_class = random.choices(
                        classes, 
                        weights=[0.7, 0.3] if len(classes) == 2 else None
                    )[0]
                    true_label = prediction_class
                    prediction_confidence = np.random.beta(9, 2)  # Skewed towards high confidence
                else:
                    # Incorrect prediction
                    prediction_class = random.choice(classes)
                    true_label = random.choice([c for c in classes if c != prediction_class])
                    prediction_confidence = np.random.beta(2, 3)  # Skewed towards lower confidence
                    
            elif model_config["type"] == "regression":
                prediction_value = np.random.normal(100, 20)
                true_value = prediction_value + np.random.normal(0, model_config.get("mae", 10))
                prediction_confidence = np.random.beta(7, 3)
                prediction_class = None
                true_label = None
                
            else:  # ranking, anomaly_detection
                prediction_confidence = np.random.beta(6, 3)
                prediction_class = None
                true_label = None
            
            # Realistic latency with normal distribution
            inference_time = np.random.normal(
                model_config["baseline_latency"] * traffic_multiplier * performance_degradation,
                model_config["latency_std"]
            )
            inference_time = max(1, inference_time)  # Minimum 1ms
            
            # System metrics
            cpu_usage = np.random.beta(2, 5) * 100  # Beta distribution for realistic CPU usage
            memory_usage = np.random.normal(1500, 300)
            gpu_usage = np.random.beta(3, 5) * 100 if model_config.get("uses_gpu") else 0
            
            error_message = None
            
        elif scenario < 0.92:  # Degraded performance (17%)
            if model_config["type"] == "classification":
                prediction_class = random.choice(model_config.get("classes", ["unknown"]))
                true_label = random.choice(model_config.get("classes", ["unknown"]))
                prediction_confidence = np.random.beta(2, 5)  # Lower confidence
            else:
                prediction_class = None
                true_label = None
                prediction_confidence = np.random.beta(2, 6)
            
            # Slower inference
            inference_time = np.random.gamma(2, model_config["baseline_latency"])
            cpu_usage = np.random.beta(5, 2) * 100  # Higher CPU
            memory_usage = np.random.normal(3000, 500)
            gpu_usage = np.random.beta(6, 2) * 100 if model_config.get("uses_gpu") else 0
            error_message = None
            
        elif scenario < 0.98:  # Slow queries (6%)
            prediction_class = None
            true_label = None
            prediction_confidence = random.uniform(0.3, 0.6)
            inference_time = np.random.exponential(model_config["baseline_latency"] * 10)
            cpu_usage = np.random.beta(7, 2) * 100
            memory_usage = np.random.normal(4000, 1000)
            gpu_usage = np.random.beta(8, 2) * 100 if model_config.get("uses_gpu") else 0
            error_message = None
            
        else:  # Errors (2%)
            prediction_class = None
            true_label = None
            prediction_confidence = 0
            inference_time = random.uniform(5000, 30000)  # Timeout scenarios
            cpu_usage = random.uniform(85, 100)
            memory_usage = random.uniform(4000, 8192)
            gpu_usage = random.uniform(90, 100) if model_config.get("uses_gpu") else 0
            error_message = random.choice(self.error_messages)
        
        # Build comprehensive log entry
        log_entry = {
            "timestamp": timestamp_str,
            "model_name": model_name,
            "model_version": model_version,
            "type": "model_inference",
            "prediction_confidence": round(float(prediction_confidence), 4),
            "inference_time_ms": round(float(inference_time), 2),
            "cpu_usage": round(float(cpu_usage), 2),
            "memory_usage_mb": round(float(memory_usage), 2),
            "gpu_usage": round(float(gpu_usage), 2),
            "environment": random.choice(self.environments),
            "region": random.choice(self.regions),
            "server": random.choice(self.servers),
            "request_id": f"req_{self.fake.uuid4()[:8]}",
            "user_id": f"user_{random.randint(1, 10000)}",
            "session_id": f"sess_{self.fake.uuid4()[:8]}",
            "data_quality_score": round(random.betavariate(8, 2), 3),
            "model_load_time_ms": round(random.gammavariate(2, 5), 2),
            "preprocessing_time_ms": round(random.gammavariate(1.5, 3), 2),
            "postprocessing_time_ms": round(random.gammavariate(1, 2), 2)
        }
        
        # Add feature statistics
        if random.random() < 0.4:  # 40% of logs include feature stats
            log_entry["feature_stats"] = {
                "num_features": random.randint(10, 200),
                "missing_features": random.randint(0, 5),
                "feature_mean": round(random.gauss(0, 1), 4),
                "feature_std": round(random.gammavariate(2, 0.5), 4)
            }
        
        # Add model-specific fields
        if model_config["type"] == "classification" and prediction_class:
            log_entry["prediction_class"] = prediction_class
            log_entry["true_label"] = true_label
            
            # Add probability distribution for multi-class
            if len(model_config.get("classes", [])) > 2:
                probs = np.random.dirichlet(np.ones(len(model_config["classes"])))
                log_entry["class_probabilities"] = {
                    cls: round(float(p), 4) 
                    for cls, p in zip(model_config["classes"], probs)
                }
        
        elif model_config["type"] == "regression":
            log_entry["predicted_value"] = round(random.gauss(100, 20), 2)
            log_entry["actual_value"] = round(log_entry["predicted_value"] + random.gauss(0, 5), 2)
            log_entry["mse"] = round(random.gammavariate(2, 50), 2)
            log_entry["mae"] = round(random.gammavariate(2, 5), 2)
        
        elif model_config["type"] == "ranking":
            log_entry["mrr_score"] = round(random.betavariate(7, 3), 4)
            log_entry["ndcg_score"] = round(random.betavariate(8, 2), 4)
            log_entry["num_candidates"] = random.randint(10, 1000)
        
        elif model_config["type"] == "anomaly_detection":
            log_entry["anomaly_score"] = round(random.betavariate(2, 8), 4)
            log_entry["is_anomaly"] = log_entry["anomaly_score"] > 0.7
            log_entry["threshold"] = 0.7
        
        # Add error details
        if error_message:
            log_entry["error_message"] = error_message
            log_entry["error"] = True
            log_entry["error_code"] = random.choice(["E001", "E002", "E003", "E004", "E005"])
            log_entry["retry_count"] = random.randint(0, 3)
        
        # Add deployment metadata
        log_entry["deployment_id"] = f"deploy_{self.fake.uuid4()[:8]}"
        log_entry["model_sha"] = self.fake.sha256()[:10]
        log_entry["container_id"] = f"container_{self.fake.uuid4()[:12]}"
        
        return log_entry
    
    def send_log(self, log_entry):
        """Send log to Logstash via TCP"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.host, self.port))
            message = json.dumps(log_entry) + '\n'
            sock.send(message.encode('utf-8'))
            sock.close()
            
            # Update statistics
            self.stats['total_sent'] += 1
            model_name = log_entry.get('model_name', 'unknown')
            self.stats['models'][model_name] = self.stats['models'].get(model_name, 0) + 1
            
            return True
        except Exception as e:
            self.stats['errors'] += 1
            if self.verbose:
                print(f"{Fore.RED}Error sending log: {e}{Style.RESET_ALL}")
            return False
    
    def generate_batch(self, count=100, delay=0.01, show_progress=True):
        """Generate and send a batch of logs"""
        print(f"{Fore.GREEN}Generating {count} ML inference logs...{Style.RESET_ALL}")
        
        # Use tqdm for progress bar
        iterator = tqdm(range(count), desc="Sending logs") if show_progress else range(count)
        
        for i in iterator:
            # Select random model with weighted probability
            model_weights = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]  # Fraud detector gets more traffic
            model = random.choices(self.models, weights=model_weights[:len(self.models)])[0]
            
            # Generate log with some temporal correlation
            if i > 0 and random.random() < 0.3:  # 30% chance of burst traffic
                timestamp = datetime.utcnow() - timedelta(seconds=random.uniform(0, 5))
            else:
                timestamp = datetime.utcnow() - timedelta(seconds=random.uniform(0, 3600))
            
            log_entry = self.generate_inference_log(model, timestamp)
            
            if self.send_log(log_entry):
                if self.verbose and (i + 1) % 100 == 0:
                    self.print_sample_log(log_entry)
            
            time.sleep(delay)
        
        self.print_statistics()
    
    def generate_continuous(self, rate=10, duration=None):
        """Generate continuous stream of logs"""
        print(f"{Fore.GREEN}Generating continuous ML logs at {rate} logs/second...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop{Style.RESET_ALL}")
        
        interval = 1.0 / rate
        start_time = time.time()
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                    
                model = random.choice(self.models)
                log_entry = self.generate_inference_log(model)
                
                self.send_log(log_entry)
                
                if self.stats['total_sent'] % 100 == 0:
                    self.print_statistics()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Stopped by user{Style.RESET_ALL}")
        
        self.print_statistics()
    
    def generate_scenario(self, scenario_type="normal"):
        """Generate specific scenario patterns"""
        scenarios = {
            "normal": {"count": 100, "error_rate": 0.02},
            "high_load": {"count": 500, "delay": 0.001},
            "degradation": {"count": 200, "latency_multiplier": 3},
            "outage": {"count": 50, "error_rate": 0.8},
            "recovery": {"count": 150, "pattern": "improving"}
        }
        
        print(f"{Fore.CYAN}Generating '{scenario_type}' scenario...{Style.RESET_ALL}")
        # Implement scenario-specific logic here
        
    def print_sample_log(self, log_entry):
        """Print a formatted sample log"""
        print(f"\n{Fore.CYAN}Sample Log:{Style.RESET_ALL}")
        print(f"  Model: {log_entry['model_name']} v{log_entry['model_version']}")
        print(f"  Confidence: {log_entry['prediction_confidence']:.2%}")
        print(f"  Latency: {log_entry['inference_time_ms']:.1f}ms")
        print(f"  CPU: {log_entry['cpu_usage']:.1f}%")
    
    def print_statistics(self):
        """Print generation statistics"""
        print(f"\n{Fore.GREEN}=== Generation Statistics ==={Style.RESET_ALL}")
        
        # Prepare data for tabulation
        table_data = []
        for model, count in self.stats['models'].items():
            percentage = (count / max(self.stats['total_sent'], 1)) * 100
            table_data.append([model, count, f"{percentage:.1f}%"])
        
        print(tabulate(table_data, headers=["Model", "Logs Sent", "Percentage"], tablefmt="grid"))
        
        print(f"\n{Fore.CYAN}Total Logs Sent: {self.stats['total_sent']}{Style.RESET_ALL}")
        print(f"{Fore.RED}Errors: {self.stats['errors']}{Style.RESET_ALL}")
        
        success_rate = ((self.stats['total_sent'] - self.stats['errors']) / max(self.stats['total_sent'], 1)) * 100
        print(f"{Fore.GREEN}Success Rate: {success_rate:.1f}%{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced ML Log Generator for ELK Stack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_log_generator.py --batch 1000
  python ml_log_generator.py --continuous --rate 20
  python ml_log_generator.py --scenario high_load
        """
    )
    
    parser.add_argument('--host', default='localhost', help='Logstash host')
    parser.add_argument('--port', type=int, default=5001, help='Logstash port')
    parser.add_argument('--batch', type=int, help='Generate batch of N logs')
    parser.add_argument('--continuous', action='store_true', help='Generate continuous logs')
    parser.add_argument('--rate', type=int, default=10, help='Logs per second for continuous mode')
    parser.add_argument('--duration', type=int, help='Duration in seconds for continuous mode')
    parser.add_argument('--delay', type=float, default=0.01, help='Delay between logs in batch mode')
    parser.add_argument('--scenario', choices=['normal', 'high_load', 'degradation', 'outage', 'recovery'],
                       help='Generate specific scenario')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    
    args = parser.parse_args()
    
    # Check system resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"{Fore.CYAN}System Status:{Style.RESET_ALL}")
    print(f"  CPU Usage: {cpu_percent}%")
    print(f"  Memory Available: {memory.available / (1024**3):.1f} GB")
    print()
    
    generator = EnhancedMLLogGenerator(args.host, args.port, args.verbose)
    
    if args.batch:
        generator.generate_batch(args.batch, args.delay, not args.no_progress)
    elif args.continuous:
        generator.generate_continuous(args.rate, args.duration)
    elif args.scenario:
        generator.generate_scenario(args.scenario)
    else:
        # Default: generate 500 logs
        generator.generate_batch(500, delay=0.01)

if __name__ == "__main__":
    main()
