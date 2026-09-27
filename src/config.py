from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TILES_DIR = DATA_DIR / "tiles"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Change these values for your study area.
PROJECT_ID = "YOUR_GOOGLE_CLOUD_PROJECT_ID"
CITY_NAME = "istanbul"
BBOX = [28.70, 40.90, 29.30, 41.20]  # min_lon, min_lat, max_lon, max_lat
START_DATE = "2024-06-01"
END_DATE = "2024-08-31"
MAX_CLOUD_PERCENT = 30
SCALE_METERS = 10
BANDS = ["B2", "B3", "B4", "B8", "NDVI"]
TILE_SIZE = 128
STRIDE = 128
RANDOM_STATE = 42
