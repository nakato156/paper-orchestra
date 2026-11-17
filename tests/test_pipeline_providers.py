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

"""Live check that the pipeline really runs on each configured provider.

Runs the outline agent -- the first step of the pipeline, and the one whose output
every later agent consumes -- end to end against each provider you have credentials
for, then asserts the outline satisfies the contract the downstream agents rely on.

    python tests/test_pipeline_providers.py

This calls real APIs and costs a few cents per provider. A provider with no
credentials configured is skipped, not failed, so the suite is useful whether you
run Gemini only, an OpenAI-compatible endpoint only, or both.

Models default to cheap, widely available ones and can be overridden:

    TEST_OPENAI_MODEL=openai:anthropic/claude-sonnet-4.5
    TEST_GEMINI_MODEL=gemini:gemini-3.6-flash

Note these are deliberately *not* the pipeline defaults in paper_writing_cli.py:
a test should exercise the plumbing with a cheap model, and the defaults are not
available on every account.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_backend_utils import resolve_model  # noqa: E402
from methods.agents.outline_agent import OutlineAgent  # noqa: E402

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")
TEMPLATE_DIR = os.path.join(REPO_DIR, "templates", "cvpr2025")

# Keys the later agents read off the outline; see literature_review_agent.py and
# plotting_agent.py. An outline missing one of these breaks the run further down.
REQUIRED_OUTLINE_KEYS = ("plotting_plan", "intro_related_work_plan", "section_plan")


def has_gemini_credentials() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or (os.environ.get("VERTEX_AI_PROJECT") and os.environ.get("VERTEX_AI_LOCATION"))
    )


def has_openai_credentials() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def run_outline(model_name: str) -> dict:
    """Runs the real outline agent and returns its parsed outline."""
    agent = OutlineAgent(model_name=model_name, cutoff_date="2025-06")
    return agent.run(
        idea_file=os.path.join(FIXTURES_DIR, "idea_sparse.md"),
        experimental_log_file=os.path.join(FIXTURES_DIR, "experimental_log.md"),
        latex_template_file=os.path.join(TEMPLATE_DIR, "template.tex"),
        conference_guidelines_file=os.path.join(TEMPLATE_DIR, "guidelines.md"),
    )


def check_outline(outline, model_name: str) -> None:
    assert isinstance(outline, dict), f"{model_name}: expected a dict, got {type(outline)}"
    missing = [key for key in REQUIRED_OUTLINE_KEYS if key not in outline]
    assert not missing, f"{model_name}: outline is missing {missing}; got {list(outline)}"
    for key in ("plotting_plan", "section_plan"):
        assert isinstance(outline[key], list), f"{model_name}: '{key}' should be a list"
        assert outline[key], f"{model_name}: '{key}' came back empty"


def main() -> int:
    # (label, env var holding a model override, default model, credentials predicate)
    legs = [
        (
            "OpenAI-compatible (OPENAI_BASE_URL / OpenRouter / OpenAI)",
            "TEST_OPENAI_MODEL",
            "openai:gpt-4o-mini",
            has_openai_credentials,
            "set OPENAI_API_KEY (and OPENAI_BASE_URL for a non-OpenAI endpoint)",
        ),
        (
            "Gemini",
            "TEST_GEMINI_MODEL",
            "gemini:gemini-3.5-flash",
            has_gemini_credentials,
            "set GEMINI_API_KEY, or VERTEX_AI_PROJECT and VERTEX_AI_LOCATION",
        ),
    ]

    ran, skipped, failed = [], [], []

    for label, env_var, default_model, has_credentials, how_to_enable in legs:
        model_name = os.environ.get(env_var, default_model)
        print(f"\n{'=' * 70}\n{label}\n  model: {model_name}\n{'=' * 70}")

        if not has_credentials():
            print(f"  SKIP - no credentials. To enable, {how_to_enable}.")
            skipped.append(label)
            continue

        provider, api_model_name = resolve_model(model_name)
        print(f"  resolve_model -> provider={provider!r}, sent to API={api_model_name!r}")

        started = time.time()
        try:
            outline = run_outline(model_name)
            check_outline(outline, model_name)
        except Exception as error:
            print(f"  FAIL after {time.time() - started:.1f}s: {type(error).__name__}: {error}")
            if "NOT_FOUND" in str(error) or "404" in str(error):
                print(
                    f"  HINT: your key cannot reach {api_model_name!r}. Model availability\n"
                    f"        varies by account tier and age, so the pipeline defaults are not\n"
                    f"        a safe assumption. List what you can reach with:\n"
                    f"          from utils.gemini_utils import get_genai_client\n"
                    f"          print([m.name for m in get_genai_client().models.list()])\n"
                    f"        then set {env_var} to one of them."
                )
            failed.append(label)
            continue

        print(
            f"  PASS in {time.time() - started:.1f}s - "
            f"{len(outline['section_plan'])} sections, "
            f"{len(outline['plotting_plan'])} figures planned"
        )
        print(f"  outline: {json.dumps(outline, ensure_ascii=False)[:200]}...")
        ran.append(label)

    print(f"\n{'=' * 70}")
    print(f"passed: {len(ran)}  failed: {len(failed)}  skipped: {len(skipped)}")
    for label in failed:
        print(f"  FAILED: {label}")
    for label in skipped:
        print(f"  skipped: {label}")

    if failed:
        return 1
    if not ran:
        print("\nNo provider was exercised. Configure at least one and run again.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
