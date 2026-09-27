import numpy as np
from src.train_rf import features


def test_feature_shape():
    tile = np.ones((8, 8, 5), dtype=np.float32)
    assert features(tile).shape == (10,)


def test_feature_values():
    tile = np.ones((4, 4, 5), dtype=np.float32)
    result = features(tile)
    assert np.allclose(result[:5], 1.0)
    assert np.allclose(result[5:], 0.0)
