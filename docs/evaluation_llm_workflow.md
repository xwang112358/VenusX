# Evaluation LLM Workflow Guide

This guide explains how the fragment-level LLM benchmark works in practice.

It is meant to answer four everyday questions:

1. What happens when I run the benchmark?
2. Which files own each part of the workflow?
3. What are the current experiment settings?
4. Where should I edit the code if I want to change something?

## 1. Big Picture

The current `evaluation_llm` module is a simple single-label classification pipeline for fragment-level InterPro prediction.

The flow is:

1. Choose a dataset such as `VenusX_Res_Act_MF50`.
2. Load fragment examples from the CSV split file.
3. Load the full InterPro label catalog from the matching `*_des.json`.
4. Build one prompt per example using the fragment and the full catalog.
5. Send the prompt to a model backend.
6. Parse the model response into normalized InterPro accessions.
7. Compute metrics.
8. Save artifacts for inspection.

The benchmark is intentionally narrower than the original design:

- only fragment-level tasks
- only active-site and binding-site tracks
- only full-catalog prompting
- only `E0` to `E3`
- no retrieval reranking in the current version

## 2. The Main Entry Points

There are two run scripts:

- `python -m evaluation_llm`
- `python -m evaluation_llm.run_openrouter_model_set`

Use `python -m evaluation_llm` when you want to run one model on one dataset with one experiment setting.

Use `python -m evaluation_llm.run_openrouter_model_set` when you want to run a preset list of OpenRouter models one after another.

## 3. What Happens in a Normal Run

The main runner is `evaluation_llm/run_fragment_benchmark.py`.

When you run:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E2 \
  --model_provider openrouter \
  --model_name openai/gpt-5-mini
```

the workflow is:

### Step 1: Build experiment settings

`build_experiment_settings(...)` combines:

- the chosen experiment preset
- your CLI overrides
- model settings

The result is one `ExperimentSettings` object.

The main fields are:

- `dataset_id`
- `split`
- `experiment_name`
- `label_card_style`
- `include_full_sequence`
- `model_provider`
- `model_name`
- `temperature`
- `max_examples`

### Step 2: Resolve dataset paths

`get_dataset_info(...)` in `evaluation_llm/fragment_dataset.py` checks the dataset id and maps it to:

- the CSV directory
- the matching description catalog
- the track name

For example:

- `VenusX_Res_Act_MF50` -> `data/interpro_2503/VenusX_Res_Act_MF50/`
- `VenusX_Res_Act_MF50` -> `data/interpro_2503/active_site/active_site_des.json`

### Step 3: Load the label catalog

`load_label_catalog(...)` in `evaluation_llm/label_catalog.py` reads the `*_des.json` file and creates `LabelCard` objects.

Each `LabelCard` contains:

- `accession`
- `catalog_index`
- `name`
- `label_type`
- `description`
- `go_terms`
- `literature_count`
- `short_desc`

Important detail:

- `interpro_id` is the canonical class id
- `interpro_label` is only treated as dataset-local bookkeeping

### Step 4: Load fragment examples

`load_fragment_examples(...)` in `evaluation_llm/fragment_dataset.py` reads:

- `train.csv`
- `valid.csv`
- or `test.csv`

and converts each row into a `FragmentExample`.

Each example keeps:

- `uid`
- `interpro_id`
- `interpro_label`
- `seq_fragment_raw`
- `fragment_parts`
- `seq_full`
- `start_parts`
- `end_parts`
- `is_multi_fragment`

Multi-part fragments are handled by splitting:

- `seq_fragment`
- `start`
- `end`

on `|`.

### Step 5: Check dataset/catalog alignment

`summarize_catalog_alignment(...)` verifies that:

- every `interpro_id` exists in the catalog
- each local `interpro_label` is internally consistent

It also reports how often `interpro_label` matches the label catalog index.

This summary is written into the run artifacts so you can inspect data quality without mixing it into the model metrics.

### Step 6: Build the model backend

`create_model_backend(...)` in `evaluation_llm/model_backends.py` selects one of:

- `mock`
- `replay`
- `openrouter`
- `agent`

Current practical backends are:

- `mock`: for smoke tests and local checks
- `replay`: for scoring saved outputs
- `openrouter`: for hosted LLM calls

The `agent` backend is a placeholder for later work.

### Step 7: Build the prompt

For every example, `build_fragment_prompt(...)` in `evaluation_llm/prompt_and_parse.py` creates a prompt with:

- task instructions
- the query uid
- the fragment parts and residue ranges
- optional full-sequence context
- the full label catalog rendered as candidate labels
- the required JSON output schema with up to 3 ranked `top_ids`

Prompt differences across experiments are small and controlled.

### Step 8: Call the model

The backend receives:

- the prompt
- the current example
- the sorted label cards

and returns a `ModelResponse`.

For OpenRouter runs:

- `.env` is loaded automatically if present
- `OPENROUTER_API_KEY` is required
- the benchmark sends one chat completion request per example

### Step 9: Parse the response

`parse_model_response(...)` tries to normalize the model output into:

- `top_ids`
- `reasoning_summary`
- `abstain`
- `parse_success`
- `invalid_labels`

`top_ids` is a ranked list with at most 3 candidate accessions.

`reasoning_summary` is a short rationale used for qualitative inspection, not a full chain-of-thought trace.

The parser is intentionally forgiving:

- it first tries to extract a JSON object
- if that fails, it falls back to scanning for `IPR...` accessions
- it can also resolve exact label names back to a unique accession

This keeps evaluation robust against small formatting mistakes.

### Step 10: Update metrics and save records

Each example becomes an `ExampleResult`.

If one backend call fails, the runner records that example as an error and continues with the rest of the dataset. This is helpful for paid API runs because a single network or provider issue does not kill the full benchmark job.

`FragmentBenchmarkMetrics` then updates:

- `main_paper_table`
- `supplemental_llm_table`

`main_paper_table` contains:

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `mcc`

`supplemental_llm_table` contains:

- `top3_acc`
- `parse_success_rate`
- `invalid_label_rate`
- `abstain_rate`
- `coverage`
- `selective_accuracy`

Finally the run writes artifacts to:

```text
artifacts/evaluation_llm/<dataset_id>/<run_name>/
```

The main files in a run directory are:

- `resolved_config.json`
- `metrics.json`
- `records.jsonl`
- `errors.jsonl`

## 4. Experiment Presets

The current experiment ladder is very small.

### `E0`

Smoke test.

- label style: `name_only`
- no full sequence
- default model: `mock/oracle`
- default max examples: `10`

### `E1`

Full catalog with only accession and name.

- label style: `name_only`
- no full sequence

### `E2`

Full catalog with accession, name, and short description.

- label style: `short_desc`
- no full sequence

### `E3`

Same as `E2`, plus full-sequence context with fragment tags.

- label style: `short_desc`
- full sequence included

## 5. Suite Mode

If you run with `--suite`, the benchmark does a small validation-selection workflow.

The suite logic is:

1. Start from an `MF50` dataset.
2. Run `E1`, `E2`, and `E3` on the validation split.
3. Pick the best setting by:
   - `accuracy`
   - then `macro_f1`
   - then `mcc`
   - then `macro_precision`
   - then `macro_recall`
4. Freeze the winning setup.
5. Evaluate that frozen setup on:
   - `MF50`
   - `MF70`
   - `MF90`

This is useful when you want a consistent comparison across similarity splits without retuning each one separately.

## 6. OpenRouter Model-Set Runs

`evaluation_llm/run_openrouter_model_set.py` is a thin wrapper around the main runner.

It:

1. builds one base experiment config
2. loads a named model set from `evaluation_llm/model_sets.py`
3. runs the benchmark once per model
4. prints a JSON summary with metrics and run directories

Current model sets:

- `starter`

This helper is useful when you want a paper-style baseline table from a fixed list of models.

## 7. The Files You Will Most Likely Edit

If you want to change a specific part of the workflow, this is the shortest path:

### Change dataset support

Edit `evaluation_llm/fragment_dataset.py`.

Typical reasons:

- add a new track
- change dataset id parsing
- change CSV field handling

### Change label-card construction

Edit `evaluation_llm/label_catalog.py`.

Typical reasons:

- change how descriptions are cleaned
- change how `short_desc` is built
- add new catalog metadata

### Change prompt wording or output instructions

Edit `evaluation_llm/prompt_and_parse.py`.

Typical reasons:

- change the system/task wording
- add extra context to prompts
- change the required JSON format

### Change how outputs are normalized

Edit `evaluation_llm/prompt_and_parse.py`.

Typical reasons:

- support a new model output style
- tighten or loosen parser behavior
- change abstention handling

### Change model backends

Edit `evaluation_llm/model_backends.py`.

Typical reasons:

- add a new API provider
- change OpenRouter request parameters
- improve mock behavior

### Change metrics

Edit `evaluation_llm/metrics.py`.

Typical reasons:

- add new slice reports
- change primary metrics
- add calibration or cost reporting

### Change experiment presets or suite behavior

Edit `evaluation_llm/run_fragment_benchmark.py`.

Typical reasons:

- add a new experiment preset
- change suite selection logic
- change artifact writing

## 8. The Most Important Data Objects

These live in `evaluation_llm/records.py`.

### `ExperimentSettings`

The run configuration after CLI parsing and preset resolution.

### `DatasetInfo`

The resolved paths and dataset metadata.

### `FragmentExample`

One evaluation example loaded from a CSV row.

### `LabelCard`

One InterPro candidate label loaded from `des.json`.

### `ModelResponse`

The raw output from the backend.

### `Prediction`

The normalized prediction after parsing.

### `ExampleResult`

The complete per-example record used for metrics and artifact writing.

## 9. Common Commands

Smoke test:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E0
```

Run a single OpenRouter model:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E2 \
  --model_provider openrouter \
  --model_name openai/gpt-5-mini
```

Run the validation-selection suite:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E2 \
  --model_provider openrouter \
  --model_name openai/gpt-5-mini \
  --suite
```

Run a preset OpenRouter model pack:

```bash
python -m evaluation_llm.run_openrouter_model_set \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E2 \
  --model_set starter
```

Run tests:

```bash
python -m unittest tests.test_evaluation_llm
```

## 10. A Simple Mental Model

If the framework feels complicated, use this shorter mental model:

```text
CSV example
  -> Label catalog
  -> Prompt
  -> Model backend
  -> Parsed prediction
  -> Metrics
  -> Saved artifacts
```

That is the whole current system.

Most of the code exists only to keep these six steps explicit and easy to debug.
