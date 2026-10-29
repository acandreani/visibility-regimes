import time
import numpy as np

def graph_features(windows, length=8, count=4, position='last'):
    """Append reversed last node strengths from a fixed, label-free NVG.

    windows: samples x time x channels. Strength is the SUM of incident
    signed vertical distances x[j]-x[i] for i<j, matching ts2vg v_distance.
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3 or not np.isfinite(windows).all():
        raise ValueError('Expected finite samples x time x channels')
    if length < count or count < 1 or windows.shape[1] < count:
        raise ValueError('Insufficient graph length')
    if position not in ('first', 'last'):
        raise ValueError('position must be first or last')
    local = windows[:, :length] if position == 'first' else windows[:, -length:]
    strengths = np.zeros_like(local)
    for i in range(local.shape[1]):
        for j in range(i + 1, local.shape[1]):
            visible = np.ones((len(local), local.shape[2]), dtype=bool)
            for k in range(i + 1, j):
                ceiling = local[:, i] + (local[:, j] - local[:, i]) * (k - i) / (j - i)
                visible &= local[:, k] < ceiling
            weight = (local[:, j] - local[:, i]) * visible
            strengths[:, i] += weight
            strengths[:, j] += weight
    return np.concatenate([windows, strengths[:, -count:][:, ::-1]], axis=1)

def examples(slices, window, horizon, trim=None):
    """Reproduce calcLellepFeatures ordering: positives, then negatives."""
    positive, negative = ([], [])
    for item in slices:
        series = np.asarray(item['series'], dtype=np.float64).T
        if trim is not None:
            series = series[-trim:]
        stop = len(series) - window - horizon
        if stop < 0 or len(series) <= window:
            continue
        negative.extend((series[i:i + window] for i in range(stop)))
        positive.append(series[stop:stop + window])
    if not positive or not negative:
        raise ValueError('Both classes required')
    return (np.array(positive + negative), np.array([1] * len(positive) + [0] * len(negative)))

def fold_slices(slices, fold, folds=3):
    train_size = int(len(slices) / (1 + folds * 0.1))
    test_size = int(train_size * 0.1)
    start = test_size * fold
    return (slices[start:start + train_size], slices[start + train_size:start + train_size + test_size])

def train_predict(x_train, y_train, x_test, epochs, batch_size, seed, device='cpu', kernel_size=3):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    conv_length = (x_train.shape[1] - 2 * (kernel_size - 1)) // 2
    model = nn.Sequential(nn.Conv1d(x_train.shape[2], 64, kernel_size), nn.ReLU(), nn.Conv1d(64, 64, kernel_size), nn.ReLU(), nn.Dropout(0.5), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(64 * conv_length, 100), nn.ReLU(), nn.Linear(100, 1)).to(device)
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.tensor(x_train.transpose(0, 2, 1), dtype=torch.float32, device=device)
    yt = torch.tensor(y_train[:, None], dtype=torch.float32, device=device)
    start = time.monotonic()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for batch in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'  epoch {epoch + 1}/{epochs}, elapsed {time.monotonic() - start:.1f}s', flush=True)
    model.eval()
    with torch.no_grad():
        test = torch.tensor(x_test.transpose(0, 2, 1), dtype=torch.float32, device=device)
        probs = torch.cat([model(batch).sigmoid() for batch in test.split(512)]).cpu().numpy().ravel()
    return (probs, sum((p.numel() for p in model.parameters())))

def metrics(y, pred):
    y, pred = (np.asarray(y), np.asarray(pred))
    if y.shape != pred.shape or not np.isin(y, [0, 1]).all() or (not np.isin(pred, [0, 1]).all()):
        raise ValueError('Invalid binary predictions or labels')
    f1 = []
    for label in (0, 1):
        denominator = np.sum(y == label) + np.sum(pred == label)
        f1.append(float(2 * np.sum((y == label) & (pred == label)) / denominator) if denominator else 0.0)
    return {'macro_f1': float(np.mean(f1)), 'positive_f1': f1[1], 'n': len(y), 'positives': int(np.sum(y == 1))}

def event_free_slices(embedded, threshold):
    """Mirror calcSlices: strict below-threshold, length > 1, no trailing slice."""
    slices = []
    start = 0
    for end in np.flatnonzero(embedded[:, 1] >= threshold):
        if end - start > 1:
            slices.append(dict(series=embedded[start:end].T.tolist(), start=int(start), stop=int(end)))
        start = int(end) + 1
    return slices
