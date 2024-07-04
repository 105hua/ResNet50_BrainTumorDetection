# This script provides an alternative method for downloading the dataset.
# Note that this script should only be used if the download_dataset.py
# script fails to download the dataset properly. The difference between
# the two scripts is that this script uses git to clone the repository.

# PLEASE ENSURE YOU HAVE GIT INSTALLED ON YOUR SYSTEM BEFORE RUNNING THIS SCRIPT.

import os
import shutil

from subprocess import call

call("git lfs install")
call("git clone https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor")

# Moves timri directory into the current directory.
dataset_dir = os.path.join(os.getcwd(), "MRI-Images-of-Brain-Tumor")
timri_dir = os.path.join(dataset_dir, "timri")
new_dataset_dir = os.path.join(os.getcwd(), "timri")
shutil.move(timri_dir, new_dataset_dir)