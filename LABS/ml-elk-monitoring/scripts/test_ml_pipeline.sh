#!/bin/bash

echo "========================================="
echo "ML Pipeline Test Suite"
echo "========================================="

# Activate virtual environment
source venv/bin/activate

# Test 1: Connection Test
echo -e "\n[TEST 1] Testing connection to Logstash..."
if python scripts/ml_log_generator.py --batch 1 --no-progress 2>/dev/null; then
    echo "✓ Connection successful"
else
    echo "✗ Connection failed - Check if Logstash is running on port 5001"
    exit 1
fi

# Test 2: Generate test batch
echo -e "\n[TEST 2] Generating test batch (50 logs)..."
python scripts/ml_log_generator.py --batch 50 --no-progress

# Wait for indexing
sleep 3

# Test 3: Verify data in Elasticsearch
echo -e "\n[TEST 3] Checking Elasticsearch..."
count=$(curl -s "localhost:9200/ml-metrics-*/_count" 2>/dev/null | grep -o '"count":[0-9]*' | cut -d':' -f2)
if [ "$count" -gt 0 ] 2>/dev/null; then
    echo "✓ Documents indexed: $count"
else
    echo "✗ No documents found in Elasticsearch"
fi

# Test 4: Check different model types
echo -e "\n[TEST 4] Checking model diversity..."
models=$(curl -s "localhost:9200/ml-metrics-*/_search?size=0" -H 'Content-Type: application/json' -d'{
  "aggs": {
    "models": {
      "cardinality": {
        "field": "model_name"
      }
    }
  }
}' 2>/dev/null | grep -o '"value":[0-9]*' | head -1 | cut -d':' -f2)

echo "✓ Unique models found: $models"

# Test 5: Check error rate
echo -e "\n[TEST 5] Checking error logs..."
errors=$(curl -s "localhost:9200/ml-metrics-*/_count" -H 'Content-Type: application/json' -d'{
  "query": {"term": {"error": true}}
}' 2>/dev/null | grep -o '"count":[0-9]*' | cut -d':' -f2)

total=$(curl -s "localhost:9200/ml-metrics-*/_count" 2>/dev/null | grep -o '"count":[0-9]*' | cut -d':' -f2)

if [ "$total" -gt 0 ] 2>/dev/null; then
    error_rate=$(echo "scale=2; $errors * 100 / $total" | bc)
    echo "✓ Error rate: ${error_rate}% ($errors/$total)"
fi

# Test 6: Performance check
echo -e "\n[TEST 6] Checking inference performance..."
avg_latency=$(curl -s "localhost:9200/ml-metrics-*/_search?size=0" -H 'Content-Type: application/json' -d'{
  "aggs": {
    "avg_latency": {
      "avg": {
        "field": "inference_time_ms"
      }
    }
  }
}' 2>/dev/null | grep -o '"value":[0-9.]*' | head -1 | cut -d':' -f2)

echo "✓ Average inference latency: ${avg_latency}ms"

echo -e "\n========================================="
echo "Test Summary:"
echo "- Total documents: $count"
echo "- Unique models: $models"
echo "- Error rate: ${error_rate}%"
echo "- Avg latency: ${avg_latency}ms"
echo "========================================="
