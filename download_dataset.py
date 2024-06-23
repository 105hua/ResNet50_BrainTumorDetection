from huggingface_hub import snapshot_download

snapshot_download(
    "PranomVignesh/MRI-Images-of-Brain-Tumor",
    repo_type="dataset",
    local_dir="./"
)