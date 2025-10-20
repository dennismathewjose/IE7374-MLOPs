#!/usr/bin/env python3
import requests
import json
from datetime import datetime
import random

base_url = "http://localhost:9200"
index_name = "ml-metrics-2025.10.20"

models = ["fraud_detector", "image_classifier", "price_predictor", "sentiment_analyzer"]

print("Inserting 50 documents...")
for i in range(50):
    doc = {
        "@timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": random.choice(models),
        "model_version": "2.3.1",
        "type": "model_inference",
        "inference_time_ms": random.randint(10, 100),
        "prediction_confidence": round(random.uniform(0.6, 0.99), 2),
        "cpu_usage": random.randint(20, 80),
        "memory_usage_mb": random.randint(500, 3000),
        "environment": "production",
        "prediction_class": "legitimate",
        "true_label": "legitimate",
        "request_id": f"manual-{i}"
    }
    
    response = requests.post(
        f"{base_url}/{index_name}/_doc",
        headers={"Content-Type": "application/json"},
        data=json.dumps(doc)
    )
    
    if response.status_code in [200, 201]:
        print(".", end="", flush=True)
    else:
        print("X", end="", flush=True)

print("\nDone!")
