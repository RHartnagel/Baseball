"""Shared HTTP session with retry/backoff and a simple per-host rate limiter."""
import time
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "Mozilla/5.0 (compatible; baseball-research/0.1)"
_MIN_INTERVAL = {"baseballsavant.mlb.com": 1.0, "statsapi.mlb.com": 0.2}
_last_call = {}
_lock = threading.Lock()

_session = None


def session():
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(total=5, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",))
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
        _session = s
    return _session


def _throttle(url):
    host = requests.utils.urlparse(url).netloc
    interval = _MIN_INTERVAL.get(host, 0.0)
    with _lock:
        wait = _last_call.get(host, 0.0) + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def get(url, params=None, timeout=120):
    """GET with throttling; raises on HTTP errors."""
    _throttle(url)
    r = session().get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r
