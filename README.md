# VenusX

VenusX is a fine-grained protein benchmark with residue-, fragment-, and domain-level annotations.
This repo now includes a standalone fragment-level LLM benchmarking module in `evaluation_llm/` for testing general LLMs on InterPro label selection.

## Setup with uv (Python 3.12)

Use `uv` to create a Python 3.12 environment and install the project dependencies:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```


## Fragment-Level LLM Benchmark

For a code-oriented walkthrough of the evaluation pipeline, see `docs/evaluation_llm_workflow.md`.

### 1. What are we doing here?

The new fragment-level benchmark evaluates whether a general LLM can assign the correct InterPro label to a protein fragment.

The benchmark currently targets:

- `VenusX_Res_Act_MF50/MF70/MF90`
- `VenusX_Res_BindI_MF50/MF70/MF90`

Each evaluation example is built directly from the local CSV files in `data/interpro_2503/...`:

- `seq_fragment` is the fragment input.
- `seq_full`, `start`, and `end` provide optional full-sequence context and explicit fragment ranges.
- `interpro_id` is the canonical gold label.
- `interpro_label` is treated as dataset-local bookkeeping only.

The label space comes from the matching `*_des.json` file:

- `data/interpro_2503/active_site/active_site_des.json`
- `data/interpro_2503/binding_site/binding_site_des.json`

Each label card contains an InterPro accession, name, cleaned description, GO terms, literature count, and a deterministic short description used for prompting.

The reduced framework is organized around a few direct files:

- `evaluation_llm/run_fragment_benchmark.py`: CLI entrypoint and suite logic
- `evaluation_llm/fragment_dataset.py`: dataset lookup and CSV loading
- `evaluation_llm/label_catalog.py`: `des.json` loading and label normalization
- `evaluation_llm/prompt_and_parse.py`: prompt construction and response parsing
- `evaluation_llm/model_backends.py`: mock, replay, and placeholder agent backends
- `evaluation_llm/metrics.py`: metric computation
- `evaluation_llm/records.py`: simple dataclasses used across the benchmark

### 2. How do we evaluate?

The module currently keeps the experiment ladder intentionally simple:

- `E0`: smoke test with a mock model.
- `E1`: full open catalog, fragment only, `accession + name`.
- `E2`: full open catalog, fragment only, `accession + name + short_desc`.
- `E3`: same as `E2`, plus full-sequence context with fragment tags.

Prompt/output contract:

- The model must return JSON only.
- Canonical format:

```json
{"top_ids":["IPR000138"],"confidence":0.87,"abstain":false}
```

- `top_ids` may contain up to 5 ranked candidate accessions.
- `top_ids[0]` is treated as the final prediction.
- Predictions are normalized against the label catalog by InterPro accession.

Current evaluation mode:

- `full_catalog`: evaluate against the full `des.json` catalog in a single-turn classification setup.

Metrics are now reported in two groups.

Main paper table:

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `mcc`

Supplemental LLM table:

- `top3_acc`
- `top5_acc`
- `parse_success_rate`
- `invalid_label_rate`
- `abstain_rate`
- `coverage`
- `selective_accuracy`

Additional analysis:

- `per_class_precision`
- `per_class_recall`
- `per_class_f1`

Slice reports are also produced for:

- seen-in-train vs unseen-in-train labels
- single-fragment vs multi-fragment examples
- short / medium / long fragment lengths

Artifacts are written to:

```text
artifacts/evaluation_llm/<dataset_id>/<run_name>/
```

Each run saves:

- resolved config
- dataset/catalog alignment summary
- metrics summary with `main_paper_table`, `supplemental_llm_table`, and slices
- per-example records with prompts, candidates, raw responses, and parsed outputs
- error records for parse failures, invalid labels, or backend failures

### 3. Example commands

Smoke test with the built-in mock adapter:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E0
```

Full-catalog run with short descriptions:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_BindI_MF50 \
  --experiment E2 \
  --model_provider mock \
  --model_name oracle
```

Validation-selection suite on `MF50`, then frozen evaluation on `MF50/MF70/MF90`:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E2 \
  --model_provider mock \
  --model_name oracle \
  --suite
```

Replay pre-generated model outputs from a JSONL file:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_BindI_MF50 \
  --experiment E2 \
  --model_provider replay \
  --model_name path/to/responses.jsonl
```

Replay JSONL format:

```json
{"uid":"P21671","raw_text":"{\"top_ids\":[\"IPR018247\"],\"confidence\":0.91,\"abstain\":false}"}
```

Run the benchmark tests:

```bash
python -m unittest tests.test_evaluation_llm
```

## OpenRouter Setup

You can call hosted LLM APIs through OpenRouter by saving your API key in a local `.env` file and using `--model_provider openrouter`.

Create a local `.env` from the committed example:

```bash
cp .env.example .env
```

Then edit `.env` and fill in your real key:

```dotenv
OPENROUTER_API_KEY="your_openrouter_api_key"
```

That is the only field you need for normal benchmark runs.

Optional `.env` fields if you want to tune request behavior:

```dotenv
OPENROUTER_MAX_TOKENS=256
OPENROUTER_TIMEOUT_SECONDS=120
```

Notes:

- `.env` is ignored by git and should not be committed.
- `.env.example` is safe to commit and is included as a template.
- The benchmark automatically loads the repo-root `.env` file when `model_provider=openrouter`.
- `OPENROUTER_HTTP_REFERER` and `OPENROUTER_TITLE` are not required. They are only useful if you want app attribution on OpenRouter.

Example OpenRouter run:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E2 \
  --model_provider openrouter \
  --model_name openai/gpt-4.1-mini
```

The benchmark sends one prompt per fragment example to OpenRouter's chat completions API and then reuses the same parser, metrics, and artifact writing flow as the mock and replay backends.

### Suggested starter models

For a first pass, use a small but representative cross-family pack instead of jumping straight to expensive flagship models. The current `starter` set in `evaluation_llm/model_sets.py` is:

- `google/gemini-2.5-flash-lite`
- `openai/gpt-4.1-mini`
- `deepseek/deepseek-chat-v3.1`
- `meta-llama/llama-3.3-70b-instruct`
- `qwen/qwen-2.5-72b-instruct`

Why this set:

- it gives you 2 inexpensive closed-model anchors and 3 strong open-family baselines
- these model families show up often in current benchmark comparisons
- they are much cheaper than frontier flagship models while still being strong enough to make the benchmark informative

There is also an `extended` set if you want to add:

- `anthropic/claude-3.5-haiku`
- `mistralai/mistral-small-3.2-24b-instruct`

Run the whole starter pack with:

```bash
python -m evaluation_llm.run_openrouter_model_set \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E2 \
  --model_set starter
```

If you only want a very small initial paper-style table, I would start with:

- `openai/gpt-4.1-mini`
- `google/gemini-2.5-flash-lite`
- `deepseek/deepseek-chat-v3.1`
- `meta-llama/llama-3.3-70b-instruct`

## Important Files

- `evaluation_llm/run_fragment_benchmark.py`: CLI runner, experiment presets, suite mode, artifact writing
- `evaluation_llm/run_openrouter_model_set.py`: helper runner for a preset pack of OpenRouter models
- `evaluation_llm/fragment_dataset.py`: dataset lookup and fragment example construction
- `evaluation_llm/label_catalog.py`: InterPro description catalog loading and short-description building
- `evaluation_llm/prompt_and_parse.py`: prompt assembly and JSON parsing
- `evaluation_llm/model_backends.py`: mock, replay, and OpenRouter-backed model calls
- `evaluation_llm/model_sets.py`: curated starter and extended OpenRouter model packs
- `evaluation_llm/metrics.py`: metrics and slice reporting
- `evaluation_llm/records.py`: core benchmark dataclasses
- `tests/test_evaluation_llm.py`: unit and smoke coverage
