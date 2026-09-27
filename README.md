# HeatWise

Explainable urban heat-risk mapping with Sentinel-2, Landsat temperature data, Random Forest, and CNN models.

## What this repository does

1. Downloads a cloud-masked Sentinel-2 composite from Google Earth Engine.
2. Adds NDVI as a derived feature.
3. Converts GeoTIFF bands into CNN-ready tiles.
4. Trains an interpretable Random Forest baseline.
5. Trains a CNN classifier.

> Sentinel-2 is the image input. It is not, by itself, a heat-risk label. For a valid supervised project, obtain a real temperature or heat-vulnerability target, such as Landsat surface-temperature products, official measurements, or a documented vulnerability dataset.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/heatwise.git
cd heatwise
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create or select a Google Cloud Project, enable Earth Engine, and authenticate:

```bash
earthengine authenticate
```

Then edit `src/config.py` and replace `YOUR_GOOGLE_CLOUD_PROJECT_ID` with the Project ID, not the project name or number.

## Download Sentinel-2 data

```bash
python -m src.download_sentinel
```

The script creates GeoTIFF bands under `data/raw/<city>/`. For large regions, use an Earth Engine Drive export instead of the direct download method.

## Create tiles

```bash
python -m src.make_tiles \
  --input data/raw/istanbul \
  --output data/tiles/istanbul \
  --tile-size 128 \
  --stride 128
```

Each tile is a NumPy array with channels `[B2, B3, B4, B8, NDVI]`.

## Prepare labels

The included label script is a starting point. You must align the temperature raster spatially to each tile before using it for research conclusions.

```bash
python -m src.make_labels \
  --tiles data/tiles/istanbul \
  --temperature path/to/landsat_temperature.tif \
  --output data/processed/labels.csv \
  --threshold 35
```

## Train models

Random Forest baseline:

```bash
python -m src.train_rf \
  --tiles data/tiles/istanbul \
  --labels data/processed/labels.csv
```

CNN:

```bash
python -m src.train_cnn \
  --tiles data/tiles/istanbul \
  --labels data/processed/labels.csv \
  --epochs 30
```

## Test

```bash
pytest -q
python -m compileall src tests
```

## Recommended research improvements

- Split train/test by geographic region, not randomly overlapping tiles.
- Align Landsat temperature pixels and Sentinel-2 tiles with a geospatial window transform.
- Report F1, ROC-AUC, calibration, and class balance.
- Compare tabular Random Forest, CNN-only, and CNN-feature-plus-Random-Forest models.
- Use SHAP for the Random Forest and Grad-CAM for CNN explanations.
- Document source licenses, acquisition dates, cloud thresholds, CRS, and preprocessing.
- Do not commit GeoTIFFs, credentials, tokens, or service-account keys to GitHub.

## Suggested repository structure

```text
heatwise/
├── src/
│   ├── config.py
│   ├── download_sentinel.py
│   ├── make_tiles.py
│   ├── make_labels.py
│   ├── train_rf.py
│   └── train_cnn.py
├── data/                 # ignored; keep only .gitkeep files in Git
├── models/               # ignored model outputs
├── tests/
├── requirements.txt
└── README.md
```
