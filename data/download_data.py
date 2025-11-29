
"""
Download TCGA Ovarian Cancer (TCGA-OV) gene expression & clinical metadata.

Files:
  - TCGA-OV.htseq_fpkm-uq.tsv.gz (gene expression)
  - TCGA-OV.GDC_phenotype.tsv.gz (clinical metadata)

Usage:
  python download_data.py

"""

import os
import requests


# Create Directory
RAW_DIR = "./raw"
os.makedirs(RAW_DIR, exist_ok=True)

# Files & URls
FILES = {
    "gene_expression": {
        "filename": "TCGA-OV.star_fpkm.tsv.gz",
        "url": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-OV.star_fpkm.tsv.gz",
    },
    "clinical": {
        "filename": "TCGA-OV.clinical.tsv.gz",
        "url": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-OV.clinical.tsv.gz",
    },
}


def download_file(url, out_path):
    """Download a file with streaming & progress messages."""
    print(f"\nDownloading: {url}")
    print(f"Saving to:   {out_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    chunk_size = 8192

    with open(out_path, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = (downloaded / total) * 100
                    print(f"\r  {percent:6.2f}% ({downloaded}/{total} bytes)", end="")

    print("\nDownload complete.")


def main():
    print("=== TCGA-OV DATA DOWNLOAD SCRIPT ===")
    print("Files will be saved in:", RAW_DIR)

    for key, item in FILES.items():
        fname = os.path.join(RAW_DIR, item["filename"])
        if os.path.exists(fname):
            print(f"\n[SKIP] {fname} already exists.")
        else:
            download_file(item["url"], fname)

    print("\nAll downloads finished.")


if __name__ == "__main__":
    main()