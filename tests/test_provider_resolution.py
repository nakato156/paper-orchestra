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

"""Self-check for provider resolution. Run with: python tests/test_provider_resolution.py

No API keys and no network needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_backend_utils import generate_image, resolve_model  # noqa: E402
from utils.openai_utils import is_o_series_model  # noqa: E402


def check(model_name, expected, env_provider=None):
    if env_provider is None:
        os.environ.pop("LLM_PROVIDER", None)
    else:
        os.environ["LLM_PROVIDER"] = env_provider
    actual = resolve_model(model_name)
    assert actual == expected, f"{model_name!r} (LLM_PROVIDER={env_provider!r}): {actual} != {expected}"


# Backwards compatibility: the names already in use must resolve exactly as before.
check("gemini-3.1-pro-preview", ("gemini", "gemini-3.1-pro-preview"))
check("gemini-3-flash-preview", ("gemini", "gemini-3-flash-preview"))
check("gemini-3-pro-image-preview", ("gemini", "gemini-3-pro-image-preview"))
check("gpt-4o", ("openai", "gpt-4o"))
check("o3-mini", ("openai", "o3-mini"))

# Explicit prefix wins, and the prefix is stripped before the name reaches the API.
check("openai:anthropic/claude-sonnet-4.5", ("openai", "anthropic/claude-sonnet-4.5"))
check("gemini:gemini-3.1-pro-preview", ("gemini", "gemini-3.1-pro-preview"))
check("openai:gpt-4o", ("openai", "gpt-4o"))
# A prefix overrides the env var.
check("gemini:gemini-3.1-pro-preview", ("gemini", "gemini-3.1-pro-preview"), env_provider="openai")

# OpenRouter variant suffixes are not provider prefixes and must survive untouched.
check("meta-llama/llama-3.3-70b-instruct:free", ("gemini", "meta-llama/llama-3.3-70b-instruct:free"))
check(
    "meta-llama/llama-3.3-70b-instruct:free",
    ("openai", "meta-llama/llama-3.3-70b-instruct:free"),
    env_provider="openai",
)

# The env var beats the legacy substring heuristic.
check("anthropic/claude-sonnet-4.5", ("openai", "anthropic/claude-sonnet-4.5"), env_provider="openai")
check("gemini-3.1-pro-preview", ("openai", "gemini-3.1-pro-preview"), env_provider="openai")
check("gpt-4o", ("gemini", "gpt-4o"), env_provider="gemini")
# Case and stray whitespace are tolerated.
check("anthropic/claude-sonnet-4.5", ("openai", "anthropic/claude-sonnet-4.5"), env_provider=" OpenAI ")

# An unknown provider fails loudly instead of silently picking a backend.
os.environ["LLM_PROVIDER"] = "openrouter"
try:
    resolve_model("anthropic/claude-sonnet-4.5")
    raise AssertionError("expected ValueError for an unknown LLM_PROVIDER")
except ValueError as error:
    assert "openrouter" in str(error), error
os.environ.pop("LLM_PROVIDER", None)

# Image generation routes the same way. Checked by which credential it demands, so no
# network and no key are needed.
os.environ.pop("LLM_PROVIDER", None)
for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "VERTEX_AI_PROJECT", "VERTEX_AI_LOCATION"):
    os.environ[key] = ""


def image_backend(model_name, env_provider=None):
    if env_provider is None:
        os.environ.pop("LLM_PROVIDER", None)
    else:
        os.environ["LLM_PROVIDER"] = env_provider
    try:
        generate_image(model_name=model_name, prompt="unused")
    except ValueError as error:
        return "openai" if "OPENAI_API_KEY" in str(error) else "gemini"
    raise AssertionError(f"{model_name}: expected a missing-credential error")


assert image_backend("gemini-3-pro-image-preview") == "gemini"
assert image_backend("openai:google/gemini-3-pro-image") == "openai"
assert image_backend("gemini:gemini-3-pro-image") == "gemini"
assert image_backend("gemini-3-pro-image", env_provider="openai") == "openai"
os.environ.pop("LLM_PROVIDER", None)

# o-series detection drives both the 'developer' role and dropping temperature.
assert is_o_series_model("o3-mini")
assert is_o_series_model("openai/o1-preview")
assert not is_o_series_model("gpt-4o")
assert not is_o_series_model("anthropic/claude-sonnet-4.5")

print("provider resolution: all checks passed")
