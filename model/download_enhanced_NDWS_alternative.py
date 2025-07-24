import kagglehub
import os
import shutil
from pathlib import Path


# Download latest version
# note first do pip install kagglehub
# then make sure you have a kaggle.json file in the root directory with your kaggle api key
# you can get your api key from https://www.kaggle.com/settings/account
path = kagglehub.dataset_download("rufaiyusufzakari/enhanced-and-modified-next-day-wildfire-spread")

print("Path to dataset files:", path)
# then move the files to the model/data/ directory