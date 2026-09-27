"""Train a compact CNN on .npy satellite tiles."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src import config


def load_data(tile_dir: Path, labels_csv: Path):
    labels = pd.read_csv(labels_csv)
    X = np.stack([np.load(tile_dir / name) for name in labels.tile]).astype("float32")
    # Robust per-channel scaling keeps reflectance and NDVI numerically stable.
    lo = np.percentile(X, 1, axis=(0, 1, 2), keepdims=True)
    hi = np.percentile(X, 99, axis=(0, 1, 2), keepdims=True)
    X = np.clip((X - lo) / (hi - lo + 1e-6), 0, 1)
    return X, labels.label.to_numpy(dtype="float32")


def train(tile_dir: Path, labels_csv: Path, output: Path, epochs: int):
    X, y = load_data(tile_dir, labels_csv)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X.shape[1:]),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=16, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"Saved CNN to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=config.MODELS_DIR / "cnn.keras")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train(args.tiles, args.labels, args.output, args.epochs)
