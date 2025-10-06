from app.data_loader import load_data
from app.processor import process_data
from app.S3_uploader import upload_to_s3
from app import config

def main():
    df = load_data()
    result_file = process_data(df)
    upload_to_s3(result_file)

if __name__ == "__main__":
    main()
