#!/bin/bash

# Monitor ML logs in Elasticsearch
while true; do
    clear
    echo "========================================"
    echo "ML ELK Stack Monitor - $(date)"
    echo "========================================"
    
    # Document count
    echo -e "\nDocument Count:"
    count=$(curl -s "localhost:9200/ml-metrics-*/_count" 2>/dev/null | grep -o '"count":[0-9]*' | cut -d':' -f2)
    echo "Total documents: $count"
    
    # Recent logs
    echo -e "\nRecent Logs (last 5):"
    curl -s "localhost:9200/ml-metrics-*/_search?size=5&sort=@timestamp:desc" -H 'Content-Type: application/json' -d'{"_source":["model_name","inference_time_ms","prediction_confidence","@timestamp"]}' 2>/dev/null | \
    python3 -c "import sys, json; data=json.load(sys.stdin); [print(f\"  {h['_source']['@timestamp'][:19]} | {h['_source']['model_name']:20} | {h['_source'].get('inference_time_ms',0):.1f}ms | Conf: {h['_source'].get('prediction_confidence',0):.2%}\") for h in data.get('hits',{}).get('hits',[])]" 2>/dev/null
    
    # Model distribution
    echo -e "\n Models Distribution:"
    curl -s "localhost:9200/ml-metrics-*/_search?size=0" -H 'Content-Type: application/json' -d'{
      "aggs": {
        "models": {
          "terms": {
            "field": "model_name",
            "size": 10
          }
        }
      }
    }' 2>/dev/null | \
    python3 -c "import sys, json; data=json.load(sys.stdin); [print(f\"  {b['key']:25} : {b['doc_count']:6} logs\") for b in data.get('aggregations',{}).get('models',{}).get('buckets',[])]" 2>/dev/null
    
    # Error rate
    echo -e "\n  Error Statistics:"
    errors=$(curl -s "localhost:9200/ml-metrics-*/_count" -H 'Content-Type: application/json' -d'{"query":{"term":{"error":true}}}' 2>/dev/null | grep -o '"count":[0-9]*' | cut -d':' -f2)
    if [ "$count" -gt 0 ] 2>/dev/null; then
        error_rate=$(echo "scale=2; $errors * 100 / $count" | bc 2>/dev/null || echo "0")
        echo "  Error rate: ${error_rate}% ($errors/$count)"
    fi
    
    # Performance stats
    echo -e "\n⚡ Performance Stats:"
    curl -s "localhost:9200/ml-metrics-*/_search?size=0" -H 'Content-Type: application/json' -d'{
      "aggs": {
        "avg_latency": {"avg": {"field": "inference_time_ms"}},
        "p95_latency": {"percentiles": {"field": "inference_time_ms", "percents": [95]}},
        "max_latency": {"max": {"field": "inference_time_ms"}}
      }
    }' 2>/dev/null | \
    python3 -c "import sys, json; data=json.load(sys.stdin); aggs=data.get('aggregations',{}); print(f\"  Avg latency: {aggs.get('avg_latency',{}).get('value',0):.1f}ms\"); print(f\"  P95 latency: {aggs.get('p95_latency',{}).get('values',{}).get('95.0',0):.1f}ms\"); print(f\"  Max latency: {aggs.get('max_latency',{}).get('value',0):.1f}ms\")" 2>/dev/null
    
    echo -e "\n========================================"
    echo "Refreshing every 2 seconds... (Ctrl+C to stop)"
    
    sleep 2
done
