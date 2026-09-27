"""Train an interpretable Random Forest baseline from tile-level features."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src import config


def features(tile):
    return np.concatenate([tile.mean(axis=(0, 1)), tile.std(axis=(0, 1))])


def train(tile_dir: Path, labels_csv: Path, model_path: Path):
    labels = pd.read_csv(labels_csv)
    X, y = [], []
    for row in labels.itertuples():
        tile = np.load(tile_dir / row.tile)
        X.append(features(tile)); y.append(int(row.label))
    X, y = np.asarray(X), np.asarray(y)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes in labels.csv.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, pred, zero_division=0))
    if len(np.unique(y_test)) == 2:
        print("ROC-AUC:", roc_auc_score(y_test, prob))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=config.MODELS_DIR / "random_forest.joblib")
    args = parser.parse_args()
    train(args.tiles, args.labels, args.model)
