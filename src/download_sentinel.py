"""Download a cloud-masked Sentinel-2 composite from Google Earth Engine.

Run from the repository root after setting PROJECT_ID in src/config.py:
    python -m src.download_sentinel
"""
import io
import zipfile
from pathlib import Path

import ee
import requests

from src import config


def mask_sentinel2_clouds(image):
    scl = image.select("SCL")
    valid = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)))
    return image.updateMask(valid).divide(10000).copyProperties(
        image, ["system:time_start"]
    )


def build_composite():
    region = ee.Geometry.Rectangle(config.BBOX)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(config.START_DATE, config.END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.MAX_CLOUD_PERCENT))
        .map(mask_sentinel2_clouds)
    )
    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError("No Sentinel-2 scenes matched the selected area and dates.")
    base = collection.median().clip(region).select(["B2", "B3", "B4", "B8"])
    ndvi = base.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return base.addBands(ndvi), region, count


def download():
    if config.PROJECT_ID == "YOUR_GOOGLE_CLOUD_PROJECT_ID":
        raise ValueError("Set PROJECT_ID in src/config.py before running.")
    ee.Authenticate()
    ee.Initialize(project=config.PROJECT_ID)
    image, region, count = build_composite()
    out_dir = config.RAW_DIR / config.CITY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{config.CITY_NAME}_sentinel2_{config.START_DATE}_{config.END_DATE}"
    response = requests.get(
        image.getDownloadURL({
            "name": name,
            "region": region,
            "scale": config.SCALE_METERS,
            "crs": "EPSG:4326",
            "fileFormat": "GeoTIFF",
            "formatOptions": {"cloudOptimized": True},
        }),
        timeout=900,
    )
    response.raise_for_status()
    zip_path = out_dir / f"{name}.zip"
    zip_path.write_bytes(response.content)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(out_dir)
    print(f"Downloaded {count} scenes; extracted files to {out_dir}")


if __name__ == "__main__":
    download()
