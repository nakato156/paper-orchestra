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

"""Checks the Semantic Scholar search used for literature discovery.

    python tests/test_scholar_search.py

Needs network but no API key and no LLM, so it isolates the search layer from the
rest of the pipeline.

Without SEMANTIC_SCHOLAR_API_KEY you share one heavily throttled pool with every
other anonymous caller, so the live checks may be skipped entirely. A skip is
reported loudly and is not a pass: an empty result from a throttled request must
never be mistaken for "the search returned nothing".
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Be gentler than the pipeline default: this script fires several queries back to
# back, and being throttled out would leave the live checks unexercised.
os.environ.setdefault("S2_MIN_REQUEST_INTERVAL", "3.0")

from utils.scholar_utils import (  # noqa: E402
    S2_FIELDS,
    _s2_get,
    is_date_valid,
    s2_relevance_search,
    s2_title_search,
)

skipped = []


def s2_is_reachable() -> bool:
    """True when S2 answers at all, so a throttle cannot look like an empty result."""
    return _s2_get({"query": "machine learning", "limit": 1, "fields": S2_FIELDS}) is not None


def skip(label: str, why: str):
    skipped.append(label)
    print(f"SKIP {label}: {why}")


# --- offline: the cutoff filter both searches share ---

assert is_date_valid("2020-05-01", 2020, "2021-01")
assert not is_date_valid("2022-05-01", 2022, "2021-01")
assert is_date_valid(None, 2019, "2021-01")
assert not is_date_valid(None, 2022, "2021-01")
assert is_date_valid(None, None, None), "no cutoff means everything passes"
print("cutoff filter: ok")

if not s2_is_reachable():
    print("\nS2 is not answering; every live check below is skipped.")
    print("Set SEMANTIC_SCHOLAR_API_KEY and run again to exercise them.")
    sys.exit(0)

# --- live: topical discovery ---

results = s2_relevance_search("diffusion model image super resolution", limit=10)
if not results:
    skip("relevance search", "S2 returned nothing (likely throttled)")
else:
    assert len(results) > 1, f"expected several papers, got {len(results)}"
    for record in results:
        assert record.get("title"), f"record without a title: {record}"

    citations = [r.get("citationCount") or 0 for r in results]
    assert citations == sorted(citations, reverse=True), (
        f"results should be most-cited first, got {citations}"
    )
    assert any(r.get("abstract") for r in results), "no abstract in any result"
    print(f"relevance search: ok ({len(results)} papers, top cited {citations[0]})")

# The cutoff must be honoured: the pipeline relies on it to avoid citing work published
# after the research cutoff. The cutoff is derived from an unfiltered search so this
# cannot pass vacuously: a fixed cutoff can legitimately filter out every result, which
# tells us nothing about whether the filter is wired in at all.
CUTOFF_QUERY = "support vector machine text classification"
unfiltered = s2_relevance_search(CUTOFF_QUERY, limit=20)
years = sorted(r["year"] for r in unfiltered if r.get("year"))

if len(years) < 4:
    skip("cutoff enforcement", "not enough dated results to derive a cutoff from")
else:
    cutoff_year = years[len(years) // 2]
    older_than_cutoff = sum(1 for y in years if y < cutoff_year)
    filtered = s2_relevance_search(
        CUTOFF_QUERY, limit=20, cutoff_date=f"{cutoff_year}-01"
    )

    for record in filtered:
        year = record.get("year")
        assert year is None or year < cutoff_year, (
            f"'{record.get('title')}' is from {year}, past the {cutoff_year}-01 cutoff"
        )
    assert filtered, (
        f"the {cutoff_year}-01 cutoff removed everything, but {older_than_cutoff} of "
        f"{len(years)} results predate it"
    )
    assert len(filtered) < len(unfiltered), "the cutoff removed nothing at all"
    print(
        f"cutoff honoured: ok ({len(unfiltered)} papers -> {len(filtered)} "
        f"before {cutoff_year}-01)"
    )

# --- live: confirming a known title ---

known = s2_title_search("Attention Is All You Need", year_hint=2017)
if known is None:
    skip("title search", "S2 returned nothing for a known title")
else:
    assert "attention" in known["title"].lower(), known["title"]
    print(f"title search: ok -> {known['title']} ({known.get('year')})")

    # A title that does not exist must come back as None rather than as some loosely
    # similar paper. s2_title_search returns None for "no match" and for "could not
    # ask", so confirm S2 actually answered this query before reading anything into it.
    NONSENSE = "Quantum Banana Transformers for Subaquatic Yodelling"
    if _s2_get({"query": NONSENSE, "limit": 3, "fields": S2_FIELDS}) is None:
        skip("title search rejection", "S2 did not answer, so None would prove nothing")
    else:
        nonsense = s2_title_search(NONSENSE)
        assert nonsense is None, f"expected no match, got {nonsense.get('title')}"
        print("title search rejects a non-existent paper: ok")

# --- the throttle keeps requests spaced out ---

interval = float(os.environ["S2_MIN_REQUEST_INTERVAL"])
started = time.monotonic()
for _ in range(3):
    _s2_get({"query": "graph neural network", "limit": 1, "fields": S2_FIELDS})
elapsed = time.monotonic() - started
assert elapsed >= interval * 2, (
    f"3 requests took {elapsed:.1f}s, expected at least {interval * 2:.1f}s; "
    "the throttle is not spacing them"
)
print(f"throttle: ok (3 requests took {elapsed:.1f}s at {interval}s spacing)")

if skipped:
    print(f"\nscholar search: checks passed, {len(skipped)} skipped: {', '.join(skipped)}")
else:
    print("\nscholar search: all checks passed")
