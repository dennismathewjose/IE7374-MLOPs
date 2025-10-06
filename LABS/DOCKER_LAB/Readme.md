# Data Pipeline using AWS S3 and Docker

This Lab is a Dockerized Python-based data pipeline that processes the **World Happiness Report 2021**, extracts insights, and uploads results to **Amazon S3**. It includes full CI/CD automation using **GitHub Actions** to validate and run the pipeline on every code change.

---

## Features

- Extracts Top 5 Happiest and Most Generous countries from the dataset
- Fully containerized using **Docker**
- Uploads processed CSV to **AWS S3**
- Automated testing with **Pytest**
- **GitHub Actions** pipeline to build, test, and run the project on every commit
- Output CSV is timestamped for versioning

---

---

## Run Locally with Docker

```bash
# Build the Docker image
docker build -t happiness-pipeline .

# Run the container with .env file for AWS credentials
docker run --env-file app/.env happiness-pipeline


---

AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=your-bucket-name
```

---
## Run Tests Locally
```bash
pip install -r requirements.txt
pytest tests/
```
---
## CI/CD with Github Actions
This project includes a GitHub Actions workflow that:

- Installs dependencies

- Runs pytest to validate pipeline logic

- Builds and runs the Docker container

- Uploads the result CSV to the configured S3 bucket

### Sample Output file

```bash
top5_happy_generous_20240523_154522.csv
```

