# <div align="center">🎻 PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing</div>
<div align="center">Yiwen Song<sup>1</sup>, Yale Song<sup>1</sup>, Tomas Pfister<sup>1</sup>, and Jinsung Yoon<sup>1</sup></div>
<div align="center"><sup>1</sup>Google Cloud AI Research</div>
<br><br>

PaperOrchestra is a multi-agent framework for automated AI research paper writing. It flexibly transforms unconstrained pre-writing materials (ideas and experimental logs) into submission-ready LaTeX manuscripts, including comprehensive literature synthesis and generated visuals, such as plots and conceptual diagrams.

Acting like an orchestrated team of specialized agents, it handles outline generation, literature review, section writing, content refinement, and plotting to close the gap between raw research materials and submission ready papers.

<div align="center">
  <img src="assets/overview.png" alt="PaperOrchestra Overview" width="90%"/>
</div>

## Key Features

- **Multi-Agent Pipeline**: Orchestrates specialized agents (Outline, Literature Review, Section Writing, Content Refinement, Plotting) for end-to-end paper generation.
- **Comprehensive Literature Synthesis**: Automatically searches, prioritizes, and integrates relevant literature into the manuscript.
- **Visuals Generation**: Generates plots and supports the inclusion of conceptual diagrams.
- **LaTeX Automation**: Produces submission-ready LaTeX manuscripts adhering to target conference templates.

## Installation

1.  **Environment Setup**: The project relies on a Conda environment named `paper_orchestra`. You can set up a similar environment with Python 3.11 and install the required dependencies.
    ```bash
    conda create -n paper_orchestra python=3.11
    conda activate paper_orchestra
    pip install -r requirements.txt
    ```

2.  **Environment Variables**: Create a `.env` file in the root of the project and add your API keys:
    ```env
    # Add keys if you have them, otherwise leave them empty
    # You need to specify related keys if you want to use the model
    # For example, if you want to use Gemini models, you need to specify either VERTEX_AI_PROJECT and VERTEX_AI_LOCATION, or GEMINI_API_KEY
    # If you want to use OpenAI models, you need to specify OPENAI_API_KEY
    
    SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
    OPENAI_API_KEY=your_openai_api_key
    VERTEX_AI_PROJECT=your_vertex_ai_project
    VERTEX_AI_LOCATION=your_vertex_ai_location
    GEMINI_API_KEY=your_gemini_api_key

    # if you want to run demo and send emails to yourself at the end, set these 2 fields as well
    SMTP_EMAIL=your_email
    SMTP_PASSWORD=your_password
    ```

## Choosing a Model Provider

Every `--*_model_name` argument accepts either a plain model name or an explicit
`provider:model` form. The supported providers are `openai` and `gemini`.

```bash
--writer_model_name gemini-3.1-pro-preview                    # Gemini (default)
--writer_model_name gpt-4o                                    # OpenAI
--writer_model_name openai:anthropic/claude-sonnet-4.5        # explicit provider
```

A provider is picked in this order:

1. An explicit `openai:` / `gemini:` prefix on the model name.
2. The `LLM_PROVIDER` environment variable (`openai` or `gemini`).
3. A legacy fallback: names containing `gpt`, `o1` or `o3` go to OpenAI, everything else
   to Gemini. This keeps existing model names working unchanged.

### OpenAI-compatible endpoints (OpenRouter, vLLM, ...)

The `openai` provider is not limited to OpenAI's own models. The client picks up
`OPENAI_BASE_URL` from the environment, so any OpenAI-compatible endpoint works, and the
model name is passed through untouched:

```env
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=your_openrouter_api_key
LLM_PROVIDER=openai
```

With `LLM_PROVIDER=openai` set, a bare name such as `anthropic/claude-sonnet-4.5` or
`meta-llama/llama-3.3-70b-instruct:free` is routed to that endpoint. OpenRouter's `:free`
style suffixes are not mistaken for a provider prefix.

See [Tests](#tests) for how to verify your setup.

### Figures and plotting

`--use_plotting` honours the selected provider too, through the same `provider:model`
form on its own two arguments:

```bash
--plotting_model_name openai:google/gemini-3.1-pro-preview \
--image_model_name openai:google/gemini-3-pro-image-preview
```

`--plotting_model_name` drives the text and vision steps: picking few-shot examples,
planning and styling the figure, writing the matplotlib code, critiquing the rendered
result and writing the caption. `--image_model_name` drives one thing only, rendering a
diagram as an image, and is the single step that needs a provider capable of generating
images.

That distinction is what decides whether figures work on a given setup:

*   **Plots** are matplotlib code that runs on your machine, so any text model produces
    them — including a local Ollama or vLLM.
*   **Diagrams** must be generated as an image. OpenRouter serves image models, so they
    work there. A local text-only endpoint cannot, and in that case each diagram is
    **skipped with a message and the run continues**, producing the plots.

Nothing in the pipeline requires Gemini credentials any more. Point `--image_model_name`
at `gemini:...` if you would rather render diagrams through Google's own API.

## Literature Search Backend

Finding papers is separate from the provider that writes the prose, because it needs a
source of real papers rather than a model's recollection of them.

| `LITERATURE_SEARCH_BACKEND` | Behaviour |
|---|---|
| `s2` (default) | Semantic Scholar. Works with any provider, including local Ollama or vLLM. |
| `gemini` | Gemini with Google Search grounding, the original behaviour. Needs Gemini credentials regardless of `LLM_PROVIDER`. |

Both backends feed the same enrichment step: every candidate is confirmed against
Semantic Scholar, and anything S2 cannot find, or that has no abstract, is dropped. The
bibliographic data in the output always comes from S2, never from a model, so a
hallucinated citation cannot reach the manuscript.

The `s2` backend works in three steps per search task: the model turns the task into a
few keyword queries, Semantic Scholar answers them, and the model then judges the results
against the task before any are kept. That last step is not optional padding. S2 ranks by
keyword relevance, so a broad query such as "efficient neural network architectures"
returns well-cited medical-imaging and 5G papers that citation count cannot separate out.
Measured on one review, filtering cut clearly off-topic results from 44% to 7% of the
final bibliography, and made the run faster because fewer candidates needed enrichment.

### Semantic Scholar is effectively mandatory

`SEMANTIC_SCHOLAR_API_KEY` is listed as optional above, but in practice you want it.
Anonymous callers share a single heavily throttled pool, and as of November 2026 that pool
commonly returns `429` for **every** request, even one at a time several seconds apart.
When that happens the literature review silently finds nothing, since an unconfirmed
candidate is discarded. Request a free key at
<https://www.semanticscholar.org/product/api>.

Two knobs for tuning the client, both rarely needed with a key:

*   `S2_MIN_REQUEST_INTERVAL` — seconds between requests, enforced process-wide across the
    discovery thread pool. Default `1.0`.
*   `S2_MAX_RETRIES` — attempts per request before giving up, with exponential backoff that
    honours S2's `Retry-After`. Default `5`.

3.  **LaTeX**: content refinement compiles the manuscript and reads the resulting PDF
    back, so `pdflatex` and `bibtex` must be on your `PATH`. Install TeX Live, or MiKTeX
    on Windows, and open a new terminal afterwards so the binaries are visible.

    On MiKTeX, also set package installation to automatic:

    ```bash
    initexmf --set-config-value="[MPM]AutoInstall=1"
    ```

    It ships set to "ask first", and a prompt no one can answer makes the first compile
    of a template hang until the pipeline's 30 second limit kills it. Compiling a
    template by hand once has the same warming effect, but only for that template's
    packages.

## Dataset

The dataset used for paper generation is **not included** in this code repository. It will be released separately at a later date.
Once available, you should download the dataset and place it under the `datasets/` directory.

## Usage

You can run the paper generation pipeline using the provided bash script or directly via the Python CLI.

### Using the Bash Script
```bash
./paper_writing_cli.sh
```

### Using the Python CLI directly
```bash
python paper_writing_cli.py \
  --raw_materials_dir /path_to_your_raw_materials/ \
  --latex_template_dir /path_to_your_latex_template/ \
  --output_dir /path_to_your_output_folder/ \
  --use_plotting true
```

### Running entirely through OpenRouter

With `OPENAI_BASE_URL` and `OPENAI_API_KEY` pointing at OpenRouter, and
`SEMANTIC_SCHOLAR_API_KEY` set, no Google credentials are needed:

```bash
python paper_writing_cli.py \
  --raw_materials_dir /path_to_your_raw_materials/ \
  --latex_template_dir templates/cvpr2025 \
  --output_dir /path_to_your_output_folder/ \
  --writer_model_name openai:google/gemini-3.1-pro-preview \
  --reflection_model_name openai:google/gemini-3.1-pro-preview
```

Pass the model names explicitly: the defaults are Gemini model ids resolved against
Google's API, which [may not be reachable from your
account](#default-model-names-may-not-be-reachable-from-your-account). Serving the same
Gemini model through OpenRouter, as above, sidesteps that entirely. Any other
OpenRouter model works the same way — `openai:anthropic/claude-sonnet-4.5`,
`openai:meta-llama/llama-3.3-70b-instruct` — as does a local endpoint, by pointing
`OPENAI_BASE_URL` at Ollama or vLLM and naming a model it serves.

Note that routing a Gemini model through OpenRouter does **not** bring back Google
Search grounding, which lives in the Google SDK rather than in the model. Discovery
works because of the `s2` backend, not the model behind it.

`--use_plotting` follows the same form through `--plotting_model_name` and
`--image_model_name`; see [Figures and plotting](#figures-and-plotting).

### Key Arguments
*   `--raw_materials_dir`: (Required) Directory containing the raw materials (e.g., idea and experimental log).
*   `--latex_template_dir`: (Required) Directory containing the LaTeX template (e.g., `templates/cvpr2025`).
*   `--output_dir`: Directory to output the generated paper. If not specified, an automatic folder with a timestamp is created.
*   `--use_plotting`: Enable the plotting agent workflow to generate figures from code (Default: `False`).
*   `--writer_model_name`: LLM for writer and literature agents. Accepts `provider:model` (see [Choosing a Model Provider](#choosing-a-model-provider)).
*   `--reflection_model_name`: LLM for reflection agents. Accepts `provider:model`.

### Interactive Demo

We provide an interactive Streamlit demo for the paper generation pipeline. For instructions on how to run and use it, please refer to the [README.md](frontend/README.md) in the `frontend` directory.

## Tests

There is no test framework to install; every script runs on plain Python.

```bash
python tests/test_provider_resolution.py   # offline, no API key, takes a second
python tests/test_scholar_search.py        # network, no API key, no LLM
python tests/test_pipeline_providers.py    # live, calls real APIs, a few cents
```

*   `test_provider_resolution.py` checks the provider routing rules: prefixes, the
    `LLM_PROVIDER` variable, the legacy fallback, and that OpenRouter `:free` style
    suffixes are not mistaken for a provider prefix.
*   `test_scholar_search.py` exercises the Semantic Scholar layer on its own: the cutoff
    filter, most-cited-first ordering, that a non-existent title is rejected rather than
    loosely matched, and that the throttle spaces requests out. If S2 is not answering it
    **skips loudly** rather than passing, so a throttled empty result is never mistaken for
    a genuine one.
*   `test_pipeline_providers.py` runs the outline agent — the first pipeline step, whose
    output every later agent consumes — end to end against each provider you have
    credentials for, then asserts the outline carries the keys the downstream agents
    read. A provider with no credentials is **skipped**, not failed, so the script is
    useful whether you run Gemini only, an OpenAI-compatible endpoint only, or both.
    It exits non-zero if a configured provider fails or if none was exercised.

It defaults to cheap models rather than the pipeline defaults, and both are overridable:

```bash
TEST_OPENAI_MODEL=openai:anthropic/claude-sonnet-4.5 \
TEST_GEMINI_MODEL=gemini:gemini-3.6-flash \
python tests/test_pipeline_providers.py
```

### Default model names may not be reachable from your account

The `--*_model_name` defaults in `paper_writing_cli.py` (`gemini-3.1-pro-preview`,
`gemini-3-flash-preview`, `gemini-3-pro-image-preview`) are **not available on every Gemini
account**. Which models a key can reach varies with the account's tier and age, so the
defaults are not a safe assumption.

Observed on a **free-tier** Gemini API key (no billing enabled), November 2026:

| Model | Result |
|---|---|
| `gemini-3.1-pro-preview` (pipeline default) | `404 NOT_FOUND` — "not found for API version v1, or is not supported for generateContent" |
| `gemini-3-flash-preview` (hardcoded in literature discovery) | `404 NOT_FOUND` |
| `gemini-2.5-pro`, `gemini-2.5-flash` | `404 NOT_FOUND` — "no longer available to new users" |
| `gemini-3.5-flash`, `gemini-3.6-flash` | works |
| `gemini-3.8-flash` | `503 UNAVAILABLE` — high demand, transient and unrelated to the above |

A paid key is likely to reach more of these, but this has not been confirmed — treat the
tier as the likely cause, not a verified one, and check your own key rather than assuming.

This bites later than you would expect: the outline step takes whatever `--writer_model_name`
you pass, so a run can start fine and only fail once it reaches literature discovery, which
has its model name hardcoded in `methods/agents/literature_review_agent.py`.

To list what your key can actually reach:

```python
from utils.gemini_utils import get_genai_client
print([m.name for m in get_genai_client().models.list()])
```

Then pass one of those explicitly, e.g. `--writer_model_name gemini-3.5-flash`. Simpler
still, serve the same Gemini model through OpenRouter — `openai:google/gemini-3.1-pro-preview`
— and your Google account's model list stops mattering at all.

## Project Structure

*   `autoraters/`: Contains scripts for automated evaluation.
*   `methods/`: Core logic for agents and pipeline execution.
*   `templates/`: LaTeX templates for target conferences, you can add your own template by creating a new subfolder under this directory.
*   `utils/`: Utility functions for API calls, PDF handling, etc.
*   `tests/`: Provider routing and live pipeline checks (see [Tests](#tests)).

## Citation

If you find this repo or our paper helpful, please cite it as follows:

```bibtex
@article{song2026paperorchestra,
  title={PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing},
  author={Song, Yiwen and Song, Yale and Pfister, Tomas and Yoon, Jinsung},
  journal={arXiv preprint arXiv:2604.05018},
  year={2026}
}
```

# Disclaimer

This is not an officially supported Google product.
