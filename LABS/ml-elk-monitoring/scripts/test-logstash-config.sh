#!/bin/bash

echo "Testing Logstash configuration..."

# Test using Docker
docker run --rm -it \
  -v $(pwd)/logstash/pipeline:/usr/share/logstash/pipeline \
  -v $(pwd)/logstash/config/logstash.yml:/usr/share/logstash/config/logstash.yml \
  docker.elastic.co/logstash/logstash:8.11.0 \
  logstash --config.test_and_exit

if [ $? -eq 0 ]; then
    echo "[PASS] Logstash configuration is valid"
else
    echo "[FAIL] Logstash configuration has errors"
fi
