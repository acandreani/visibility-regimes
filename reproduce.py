"""Verify all released predictions and regenerate the Figure 7 numeric grids."""
import hashlib
import json
from pathlib import Path
import numpy as np
from visibility_regimes import metrics

ROOT = Path(__file__).resolve().parent


def reproduce():
    index = json.loads((ROOT/'data/index.json').read_text())
    results = []
    for cell in index['cells']:
        scores, labels = {}, {}
        for mode in ('raw', 'vg8'):
            yy, pp, folds = [], [], []
            runs = sorted((r for r in cell['runs'] if r['mode'] == mode), key=lambda r:r['fold'])
            assert [r['fold'] for r in runs] == [0, 1, 2]
            for run in runs:
                path = ROOT/run['file']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == run['sha256'], path
                data = np.load(path, allow_pickle=False)
                y, p = data['labels'], data['probabilities']
                assert y.shape == p.shape and np.isfinite(p).all()
                assert ((p >= 0) & (p <= 1)).all()
                if mode == 'raw':
                    labels[run['fold']] = y
                else:
                    np.testing.assert_array_equal(y, labels[run['fold']])
                folds.append(metrics(y, p > .5)['macro_f1'])
                yy.extend(y); pp.extend(p)
            scores[mode] = dict(fold_mean=float(np.mean(folds)),
                                pooled=metrics(yy, np.asarray(pp) > .5)['macro_f1'])
        a, b = scores['raw'], scores['vg8']
        results.append(dict(system=cell['system'], W=cell['W'], T=cell['T'], L=cell['L'],
            scores=scores, relative_gain_percent=100*(b['pooled']/a['pooled']-1),
            fold_mean_difference_pp=100*(b['fold_mean']-a['fold_mean'])))
    return dict(cells=results, unavailable=index['unavailable'])


if __name__ == '__main__':
    result = reproduce()
    (ROOT/'output').mkdir(exist_ok=True)
    (ROOT/'output/heatmaps.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"Verified {len(result['cells'])} cells, 1512 predictions; 12 unavailable cells retained.")
