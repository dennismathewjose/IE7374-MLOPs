# TFDV MLOps Lab

This lab simulates a professional Data Validation workflow using TensorFlow Data Validation (TFDV).

## Project Structure
* **notebooks/**: Interactive environment to explore data and create the "Golden Schema".
* **pipelines/**: Python scripts that run automated validation (simulating production).
* **config/**: YAML configuration for handling column names and thresholds.
* **artifacts/**: Stores the generated schema (`schema.pbtxt`).

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate