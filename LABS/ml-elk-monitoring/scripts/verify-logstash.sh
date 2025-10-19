#!/bin/bash

echo "========================================="
echo "Logstash Verification Report"
echo "========================================="

# Check Logstash container
if docker ps | grep -q ml-logstash; then
    echo "[PASS] Logstash container is running"
else
    echo "[FAIL] Logstash container not running"
fi

# Check Logstash API
if curl -s -X GET "http://localhost:9600" > /dev/null 2>&1; then
    echo "[PASS] Logstash API is accessible"
else
    echo "[FAIL] Logstash API not accessible"
fi

# Check pipeline status
pipeline_running=$(curl -s "localhost:9600/_node/stats/pipelines" | grep -c '"events_out"')
if [ "$pipeline_running" -gt 0 ]; then
    echo "[PASS] Logstash pipeline is running"
else
    echo "[WARN] Logstash pipeline might not be processing"
fi

# Check test data in Elasticsearch
doc_count=$(curl -s "localhost:9200/ml-metrics-*/_count" 2>/dev/null | grep -o '"count":[0-9]*' | cut -d':' -f2)
if [ "$doc_count" -gt 0 ] 2>/dev/null; then
    echo "[PASS] Test data indexed successfully (count: $doc_count)"
else
    echo "[WARN] No data indexed yet"
fi

echo "========================================="
