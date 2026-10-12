"""On-disk parquet cache for DataFrames, keyed by a name + params hash."""
import hashlib
import json
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _path(name, params):
    h = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{name}_{h}.parquet"


def cached_frame(name, params, fetch):
    """Return the cached DataFrame for (name, params), calling fetch() on a miss."""
    p = _path(name, params)
    if p.exists():
        return pd.read_parquet(p)
    df = fetch()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df
