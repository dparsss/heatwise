"""Create labels.csv from a temperature raster aligned to the tile grid.

The temperature raster must be a GeoTIFF in degrees Celsius or another consistent
unit. For a real study, obtain it from Landsat thermal products or a validated
urban-temperature dataset; do not use random labels.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio


def label_tiles(tile_dir: Path, temperature_raster: Path, output_csv: Path, threshold: float):
    with rasterio.open(temperature_raster) as src:
        temp = src.read(1).astype("float32")
    rows = []
    for tile_path in sorted(tile_dir.glob("tile_*.npy")):
        tile = np.load(tile_path)
        # Placeholder spatial mapping: replace with geospatial window mapping when
        # temperature and Sentinel rasters have different grids.
        mean_temp = float(np.nanmean(temp))
        rows.append({"tile": tile_path.name, "temperature": mean_temp,
                     "label": int(mean_temp >= threshold)})
    if not rows:
        raise RuntimeError("No tiles found.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote {len(rows)} labels to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--temperature", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/labels.csv"))
    parser.add_argument("--threshold", type=float, default=35.0)
    args = parser.parse_args()
    label_tiles(args.tiles, args.temperature, args.output, args.threshold)
