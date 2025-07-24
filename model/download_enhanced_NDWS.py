import kagglehub

# Download latest version
# note first do pip install kagglehub
# then make sure you have a kaggle.json file in the root directory with your kaggle api key
# it should be in ~/.kaggle/kaggle.json (you will need to create the directory)
# you can get your api key from https://www.kaggle.com/settings/account
path = kagglehub.dataset_download("rufaiyusufzakari/enhanced-and-modified-next-day-wildfire-spread")

print("Path to dataset files:", path)