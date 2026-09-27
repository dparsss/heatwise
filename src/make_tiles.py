"""Create aligned multi-band .npy tiles from downloaded GeoTIFF files."""
import argparse
from pathlib import Path

import numpy as np
import rasterio

from src import config


def make_tiles(input_dir: Path, output_dir: Path, tile_size: int, stride: int):
    files = {p.stem: p for p in input_dir.glob("*.tif")}
    missing = [b for b in config.BANDS if b not in files]
    if missing:
        raise FileNotFoundError(f"Missing bands: {missing}; found {sorted(files)}")
    arrays = []
    profile = None
    for band in config.BANDS:
        with rasterio.open(files[band]) as src:
            arrays.append(src.read(1).astype("float32"))
            profile = src.profile
    stack = np.stack(arrays, axis=-1)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    h, w, _ = stack.shape
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = stack[y:y + tile_size, x:x + tile_size]
            if np.mean(np.any(tile != 0, axis=-1)) < 0.5:
                continue
            np.save(output_dir / f"tile_{count:05d}.npy", tile)
            count += 1
    print(f"Created {count} tiles with shape ({tile_size}, {tile_size}, {len(config.BANDS)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.RAW_DIR / config.CITY_NAME)
    parser.add_argument("--output", type=Path, default=config.TILES_DIR / config.CITY_NAME)
    parser.add_argument("--tile-size", type=int, default=config.TILE_SIZE)
    parser.add_argument("--stride", type=int, default=config.STRIDE)
    args = parser.parse_args()
    make_tiles(args.input, args.output, args.tile_size, args.stride)
