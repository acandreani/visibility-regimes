"""Label-independent visibility augmentation using only observed samples."""
import numpy as np
from .core import metrics, examples, fold_slices, event_free_slices, train_predict
from .core import graph_features as _graph_features

__version__ = '0.1.0'


def augment(windows):
    """Append four signed node strengths; shape (samples, W, channels) -> W+4.

    Uses the last min(W,8) observed samples, identically for all classes.
    No labels or future values are accepted. W must be at least four.
    """
    x = np.asarray(windows, dtype=float)
    if x.ndim != 3 or x.shape[1] < 4:
        raise ValueError('Expected (samples, W>=4, channels)')
    return _graph_features(x, length=min(x.shape[1], 8), position='last')


def fit_predict(train, labels, test, *, graph=True, epochs=100,
                batch_size=32, seed=20261022, device='cpu'):
    """Train the exploratory PyTorch CNN; returns probabilities.

    No normalization, class weighting or equal-capacity padding is applied.
    This is not the separate Keras Ikeda or equal-capacity table protocol.
    """
    x, tx, y = np.asarray(train), np.asarray(test), np.asarray(labels)
    if x.ndim != 3 or tx.ndim != 3 or x.shape[1:] != tx.shape[1:] or x.shape[1] < 4:
        raise ValueError('Train/test must share (W>=4, channels)')
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected one binary label per training window')
    if not np.isfinite(x).all() or not np.isfinite(tx).all():
        raise ValueError('Nonfinite observations')
    kernel = 2 if x.shape[1] < 6 else 3
    a, b = (augment(x), augment(tx)) if graph else (x, tx)
    return train_predict(a, y, b, epochs, batch_size, seed, device=device, kernel_size=kernel)[0]
