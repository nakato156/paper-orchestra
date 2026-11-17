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
from typing import Optional, Callable, Dict, Any, Tuple

from google.genai import types

from utils.gemini_utils import (
    call_gemini_with_text_prompt,
    parse_gemini_json_results,
    call_gemini_with_contents,
    call_gemini_with_images,
)
from utils.openai_utils import (
    call_openai_models_with_text_prompt,
    call_openai_models_with_content,
    parse_openai_json_results,
    get_openai_image_part_from_path,
)
from utils.pdf_utils import load_paper


PROVIDERS = ("openai", "gemini")


def identity_parse(response: str):
    return response


def resolve_model(model_name: str) -> Tuple[str, str]:
    """Picks the backend for a model, returning (provider, name to send to the API).

    Resolution order:
      1. An explicit "openai:" or "gemini:" prefix, e.g. "openai:anthropic/claude-sonnet-4.5".
      2. The LLM_PROVIDER environment variable.
      3. The legacy gpt/o1/o3 substring heuristic, so existing model names keep working.

    Any OpenAI-compatible endpoint (OpenRouter, vLLM, ...) is reached with the "openai"
    provider plus OPENAI_BASE_URL; the model name itself is passed through untouched.
    """
    prefix, separator, rest = model_name.partition(":")
    if separator and prefix.lower() in PROVIDERS:
        return prefix.lower(), rest

    default_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if default_provider:
        if default_provider not in PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {PROVIDERS}, got '{default_provider}'."
            )
        return default_provider, model_name

    lowered = model_name.lower()
    if "gpt" in lowered or "o1" in lowered or "o3" in lowered:
        return "openai", model_name
    return "gemini", model_name


def get_parser_for_provider(provider: str, return_json: bool = True) -> Callable:
    if not return_json:
        return identity_parse

    if provider == "openai":
        return parse_openai_json_results
    else:
        return parse_gemini_json_results


def get_llm_parser(model_name: str, return_json: bool = True) -> Callable:
    return get_parser_for_provider(resolve_model(model_name)[0], return_json)


def call_llm_with_text_prompt(
    prompt: str,
    model_name: str,
    generation_configs: Optional[Dict[str, Any]] = None,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Unified function to call either Gemini or OpenAI models based on the model_name.
    """
    provider, api_model_name = resolve_model(model_name)
    parser = (
        result_parsing_func
        if result_parsing_func is not None
        else get_parser_for_provider(provider, return_json)
    )

    if generation_configs is None:
        generation_configs = {}

    if provider == "openai":
        # Copy to avoid side-effects on caller's dictionary
        configs_for_openai = dict(generation_configs)
        system_prompt = configs_for_openai.pop("system_instruction", None)

        response_dict = call_openai_models_with_text_prompt(
            prompt=prompt,
            model_name=api_model_name,
            generation_configs=configs_for_openai,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
            system_prompt=system_prompt,
        )
        return response_dict
    else:
        response_dict = call_gemini_with_text_prompt(
            prompt=prompt,
            model_name=api_model_name,
            generation_configs=generation_configs,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
        )
        return response_dict


def call_llm_with_images(
    prompt: str,
    images: list[str],
    model_name: str,
    generation_configs: Optional[Dict[str, Any]] = None,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Unified function to call either Gemini or OpenAI models with images based on the model_name.
    """
    provider, api_model_name = resolve_model(model_name)
    parser = (
        result_parsing_func
        if result_parsing_func is not None
        else get_parser_for_provider(provider, return_json)
    )

    if generation_configs is None:
        generation_configs = {}

    if provider == "openai":
        # Copy to avoid side-effects on caller's dictionary
        configs_for_openai = dict(generation_configs)
        system_prompt = configs_for_openai.pop("system_instruction", None)

        content = [{"type": "text", "text": prompt}]
        for image_path in images:
            content.extend(get_openai_image_part_from_path(image_path))

        response_dict = call_openai_models_with_content(
            content=content,
            model_name=api_model_name,
            generation_configs=configs_for_openai,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
            system_prompt=system_prompt,
        )
        return response_dict
    else:
        response_dict = call_gemini_with_images(
            prompt=prompt,
            images=images,
            model_name=api_model_name,
            generation_configs=generation_configs,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
        )
        return response_dict


def call_llm_with_pdf(
    pdf_path: str,
    prompt: str,
    model_name: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    check_parsed_response_not_none: bool = True,
    return_json: bool = True,
    result_parsing_func: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Unified function to evaluate a PDF.
    Passes raw PDF bytes to Gemini, and extracted text to OpenAI.
    """
    provider, api_model_name = resolve_model(model_name)
    parser = (
        result_parsing_func
        if result_parsing_func is not None
        else get_parser_for_provider(provider, return_json)
    )

    if provider == "openai":
        paper_text = load_paper(pdf_path, min_size=100)
        full_prompt = f"Paper Content:\n{paper_text}\n\nTask:\n{prompt}"

        response_dict = call_openai_models_with_text_prompt(
            prompt=full_prompt,
            model_name=api_model_name,
            generation_configs={"temperature": temperature},
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
            system_prompt=system_instruction,
        )
        return response_dict
    else:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        contents = [
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=prompt),
        ]

        generation_configs = {"temperature": temperature}
        if system_instruction is not None:
            generation_configs["system_instruction"] = system_instruction

        response_dict = call_gemini_with_contents(
            contents=contents,
            model_name=api_model_name,
            generation_configs=generation_configs,
            result_parsing_func=parser,
            check_parsed_response_not_none=check_parsed_response_not_none,
        )
        return response_dict
