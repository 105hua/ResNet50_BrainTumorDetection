import os
from huggingface_hub import snapshot_download

# KNOWN ISSUE:
# Sometimes, the dataset will fail to download as a result of
# an error with downloading the files from the repository.
# Currently, there is no known workaround for this issue.
# If the dataset fails to download, please try running the
# script again. If you happen to know a fix for this issue,
# please let me know by creating an issue on the repository.

local_dir = os.getcwd()
cache_dir = os.path.join(local_dir, "cache")

try:
    snapshot_download(
        "PranomVignesh/MRI-Images-of-Brain-Tumor",
        repo_type="dataset",
        local_dir="./",
        cache_dir="./cache"
    )
except Exception as e:
    print(f"Error downloading dataset.\n{e}")

print("Dataset should now be available in the current directory.")