import kagglehub
import os
os.environ['KAGGLEHUB_CACHE'] = './data/raw/'
# Download latest version
path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")

print("Path to dataset files:", path)