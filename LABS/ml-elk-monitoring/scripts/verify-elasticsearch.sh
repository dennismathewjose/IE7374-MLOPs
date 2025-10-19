#!/bin/bash

echo "========================================="
echo "Elasticsearch Verification Report"
echo "========================================="

# Check Elasticsearch is running
if curl -s -X GET "http://localhost:9200" > /dev/null 2>&1; then
    echo "[PASS] Elasticsearch is running"
else
    echo "[FAIL] Elasticsearch is not reachable"
    exit 1
fi

# Check cluster health
health=$(curl -s -X GET "http://localhost:9200/_cluster/health" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
if [ "$health" = "yellow" ] || [ "$health" = "green" ]; then
    echo "[PASS] Cluster health is $health"
else
    echo "[FAIL] Cluster health is $health"
fi

# Check template exists
if curl -s -X GET "http://localhost:9200/_index_template/ml-metrics-template" | grep -q "ml-metrics-template"; then
    echo "[PASS] ML metrics template exists"
else
    echo "[FAIL] ML metrics template not found"
fi

# Check if test index has correct mapping
if curl -s -X GET "http://localhost:9200/ml-metrics-test/_mapping" | grep -q '"inference_time_ms":{"type":"float"}'; then
    echo "[PASS] Template mapping correctly applied"
else
    echo "[WARN] Template mapping might not be applied correctly"
fi

# Check document count in test index
doc_count=$(curl -s -X GET "http://localhost:9200/ml-metrics-test/_count" | grep -o '"count":[0-9]*' | cut -d':' -f2)
if [ "$doc_count" -gt 0 ] 2>/dev/null; then
    echo "[PASS] Test documents inserted successfully (count: $doc_count)"
else
    echo "[WARN] No test documents found"
fi

echo "========================================="
echo "Verification complete"
echo "========================================="
