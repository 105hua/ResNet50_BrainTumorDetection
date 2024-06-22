@echo off

REM Seems to be more reliable than using hugging face hub.
REM Known Issue: Sometimes downloading through hub fails due to network issues.

git lfs install
git clone https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor

echo Done!