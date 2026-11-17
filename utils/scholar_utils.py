# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import requests
import time
import threading
from typing import List, Dict, Optional
from thefuzz import fuzz  # type: ignore
import datetime

from dotenv import load_dotenv  # type: ignore

cur_dir = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(cur_dir, "../.env"))
S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
print(f"Semantic Scholar API key: {'set' if S2_API_KEY else 'not set (shared rate limit)'}")


# --- SEMANTIC SCHOLAR UTILS ---

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,venue,year,abstract,citationCount,journal,publicationDate"

# Discovery runs on a thread pool, so the request interval is enforced process-wide
# here rather than by each caller. Without an API key S2 shares a pool across all
# anonymous users and will start answering 429 well below one request per second.
_s2_request_lock = threading.Lock()
_s2_min_interval = float(os.environ.get("S2_MIN_REQUEST_INTERVAL", "1.0"))
_s2_max_retries = int(os.environ.get("S2_MAX_RETRIES", "5"))
_s2_last_request_at = 0.0


def _s2_wait_for_slot():
    """Blocks until this caller may issue its request, keeping requests spaced out."""
    global _s2_last_request_at
    with _s2_request_lock:
        pending = _s2_min_interval - (time.monotonic() - _s2_last_request_at)
        if pending > 0:
            time.sleep(pending)
        _s2_last_request_at = time.monotonic()


def _s2_get(params: Dict, max_retries: int = None) -> Optional[Dict]:
    """Single throttled exit point to the S2 search API.

    Returns the payload, or None when the request could not be completed -- which the
    callers deliberately flatten into an empty result, since a missing source should
    degrade a literature search rather than abort the run.
    """
    if max_retries is None:
        max_retries = _s2_max_retries
    # Lowercase per the S2 reference, which documents this header as case-sensitive.
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}

    for attempt in range(max_retries):
        _s2_wait_for_slot()
        try:
            rsp = requests.get(
                S2_SEARCH_URL, headers=headers, params=params, timeout=10
            )
        except Exception as e:
            print(f"      [S2] request failed for '{params.get('query')}': {e}")
            continue

        if rsp.status_code == 200:
            return rsp.json()

        if rsp.status_code == 429:
            # Anonymous callers share one pool and are throttled hard, so back off
            # generously; S2 tells us how long when it can.
            backoff = min(2**attempt, 30)
            try:
                backoff = max(backoff, int(rsp.headers.get("Retry-After", 0)))
            except (TypeError, ValueError):
                pass
            print(
                f"      [S2] rate limited, retrying in {backoff}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(backoff)
            continue

        print(f"      [S2] returned HTTP {rsp.status_code} for '{params.get('query')}'")
        return None

    print(
        f"      [S2] gave up after {max_retries} attempts for '{params.get('query')}'."
        f"{'' if S2_API_KEY else ' Set SEMANTIC_SCHOLAR_API_KEY to raise the limit.'}"
    )
    return None


def is_date_valid(p_date, p_year, cutoff_str) -> bool:
    if not cutoff_str:
        return True
    try:
        c_year, c_month = map(int, cutoff_str.split("-"))
        cutoff_dt = datetime.datetime(c_year, c_month, 1)
        if p_date:
            try:
                p_dt = datetime.datetime.strptime(p_date, "%Y-%m-%d")
                return p_dt < cutoff_dt
            except ValueError:
                pass
        if p_year:
            if p_year < c_year:
                return True
            if p_year == c_year and c_month > 1:
                return True
            return False
        return True
    except Exception:
        return True


def s2_title_search(
    title_query: str, year_hint: int = None, cutoff_date: str = None
) -> Optional[Dict]:
    """Confirms a known title. The fuzzy gate rejects anything that is not that paper."""
    payload = _s2_get(
        {"query": title_query, "limit": 3, "fields": S2_FIELDS}
    )
    if not payload:
        return None

    results = payload.get("data", [])
    if not results:
        return None

    best_match = None
    highest_ratio = 0

    for r in results:
        if not r.get("title"):
            continue
        if not is_date_valid(r.get("publicationDate"), r.get("year"), cutoff_date):
            continue

        ratio = fuzz.ratio(title_query.lower(), r["title"].lower())
        if year_hint and r.get("year") == year_hint:
            ratio += 10

        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = r

    if highest_ratio > 70:
        return best_match
    return None


def s2_relevance_search(
    query: str, limit: int = 15, cutoff_date: str = None
) -> List[Dict]:
    """Finds papers on a topic, most cited first.

    Deliberately has no fuzzy title gate: that exists in s2_title_search to confirm a
    title you already know, which is the opposite of discovery.
    """
    payload = _s2_get({"query": query, "limit": limit, "fields": S2_FIELDS})
    if not payload:
        return []

    results = [
        r
        for r in payload.get("data", [])
        if r.get("title")
        and is_date_valid(r.get("publicationDate"), r.get("year"), cutoff_date)
    ]
    results.sort(key=lambda r: r.get("citationCount") or 0, reverse=True)
    return results
