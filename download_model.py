import os
from huggingface_hub import hf_hub_download, HfApi

# Specify the model repository
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with your desired model repository
destination_dir = "/mnt/data/thuandv/LIDA/models/Meta-Llama-3-8B-Instruct"  # Local directory to save the files

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Initialize the Hugging Face API
api = HfApi()

# Get a list of all files in the repository
files = api.list_repo_files(repo_id)

# Download each file
for file_name in files:
    file_path = hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=destination_dir)
    print(f"Downloaded {file_name} to {file_path}")

print(f"All files from {repo_id} have been downloaded to {destination_dir}")
