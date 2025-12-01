import os
import sys
import yaml
import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_validation():
    # 1. Setup Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'lab_config.yaml')
    SCHEMA_PATH = os.path.join(BASE_DIR, 'artifacts', 'schema.pbtxt')

    # 2. Load Resources
    config = load_config(CONFIG_PATH)
    
    if not os.path.exists(SCHEMA_PATH):
        print(f"ERROR: Schema not found at {SCHEMA_PATH}")
        print("Please run the Notebook first to generate the Golden Schema.")
        sys.exit(1)

    print("Loading Golden Schema...")
    schema = tfdv.load_schema_text(SCHEMA_PATH)

    # 3. Simulate "New Serving Data" (with Drift)
    print("Simulating incoming serving data...")
    # We download the same data but artificially age everyone by 30 years to cause drift
    df = pd.read_csv(config['data']['train_url'], names=config['data']['column_names'], skipinitialspace=True)
    serving_df = df.sample(n=1000, random_state=99)
    serving_df['age'] = serving_df['age'] + 30 
    
    # 4. Compute Stats for New Data
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_df)

    # 5. Configure Drift Check from Config
    # We update the schema in memory to include the drift threshold defined in YAML
    drift_threshold = config['validation']['drift_threshold']
    tfdv.get_feature(schema, 'age').drift_comparator.jensen_shannon_divergence.threshold = drift_threshold

    # 6. Validate (Check for Anomalies AND Drift)
    # Note: To check drift, we need previous training stats. 
    # For simplicity in this script, we assume we only check schema compliance here,
    # but normally you would load 'train_stats.pb' as well.
    
    print("Validating data against schema...")
    anomalies = tfdv.validate_statistics(statistics=serving_stats, schema=schema)

    # 7. Report Results
    if anomalies.anomaly_info:
        print("\n!!! ANOMALIES DETECTED !!!")
        print(anomalies)
        # In MLOps, this exit code (1) would stop the deployment pipeline
        sys.exit(1)
    else:
        print("\nData Validation Passed. No anomalies.")
        sys.exit(0)

if __name__ == "__main__":
    run_validation()