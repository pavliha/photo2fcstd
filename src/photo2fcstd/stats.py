import numpy as np

RESAMPLES = 10000
SEED = 0


def mean_ci(values, resamples=RESAMPLES, seed=SEED):
    v = np.asarray(list(values), float)
    if not len(v):
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), (resamples, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_delta(before, after, resamples=RESAMPLES, seed=SEED):
    shared = [k for k in after if k in before]
    if not shared:
        return None
    d = np.array([after[k] - before[k] for k in shared], float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (resamples, len(d)))].mean(axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"delta": float(d.mean()), "lo": lo, "hi": hi, "n": len(shared), "significant": lo > 0 or hi < 0}


def resolution(values, resamples=RESAMPLES, seed=SEED):
    v = np.asarray(list(values), float)
    if len(v) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), (resamples, len(v)))].mean(axis=1)
    return float((np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2)


def parts_needed(values, effect, resamples=RESAMPLES, seed=SEED):
    have = resolution(values, resamples, seed)
    if effect <= 0 or not have:
        return 0
    return int(round(len(list(values)) * (have / effect) ** 2))
