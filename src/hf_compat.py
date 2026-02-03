# hf_compat.py
from datasets import load_dataset
import numpy as np

PARQUET_REVISION = "refs/convert/parquet"

def _load_any(dataset_name: str, **kwargs):
    """
    Load dataset (all splits) from main; if datasets>=4 complains about scripts,
    fall back to the Parquet-converted branch.
    """
    try:
        return load_dataset(dataset_name, **kwargs)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            return load_dataset(dataset_name, revision=PARQUET_REVISION, **kwargs)
        raise

def _select_case_insensitive(dd, split: str):
    split_l = split.lower()
    for k in dd.keys():
        if k.lower() == split_l:
            return dd[k]
    return None

def _pseudo_train_val_from_single_split(base_ds, seed: int = 42, val_ratio: float = 0.05):
    """
    Create deterministic train/validation from a single available split.
    - train: (1 - val_ratio)
    - validation: val_ratio
    - test: FULL base_ds (so your FAISS DB can be the whole dataset)
    """
    n = len(base_ds)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()

    train_ds = base_ds.select(train_idx)
    val_ds = base_ds.select(val_idx)

    return train_ds, val_ds, base_ds  # test == full base

def load_dataset_compat(
    dataset_name: str,
    split: str,
    *,
    seed: int = 42,
    val_ratio: float = 0.05,
    **kwargs
):
    """
    Compatible loader for:
      - datasets>=4.0.0 where dataset scripts are unsupported
      - datasets that only have a single split (e.g., only 'test')
    """
    # Try direct load first (maybe works if split exists and no script)
    try:
        return load_dataset(dataset_name, split=split, **kwargs)
    except RuntimeError as e:
        # Will handle below by loading all splits with parquet fallback
        if "Dataset scripts are no longer supported" not in str(e):
            raise
    except ValueError:
        # Unknown split; handle below
        pass

    # Load *all available splits* (main or parquet)
    dd = _load_any(dataset_name, **kwargs)  # DatasetDict
    ds = _select_case_insensitive(dd, split)
    if ds is not None:
        return ds

    # If split doesn't exist, but there is exactly one real split, create pseudo train/val
    keys = list(dd.keys())
    if len(keys) == 1 and split.lower() in {"train", "validation", "val", "test"}:
        base = dd[keys[0]]
        train_ds, val_ds, test_ds = _pseudo_train_val_from_single_split(base, seed=seed, val_ratio=val_ratio)
        if split.lower() == "train":
            return train_ds
        if split.lower() in {"validation", "val"}:
            return val_ds
        return test_ds  # "test"

    raise ValueError(f'Unknown split "{split}". Available splits: {keys}')
