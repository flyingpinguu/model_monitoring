"""Download the raw taxi data. This is the first stage of the DVC pipeline.

Run it from the project folder with:

    uv run python -m src.download_data
"""

from src.preprocess import ROOT_DIR, download_data

if __name__ == "__main__":
    path = download_data()
    print(f"Data ready at {path.relative_to(ROOT_DIR)}")
