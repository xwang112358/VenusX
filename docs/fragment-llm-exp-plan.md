# Fragment-Level LLM Experiment Plan

This document is a practical runbook for the first rounds of fragment-level LLM experiments.

It is designed for the current `evaluation_llm/` framework:

- tasks: fragment-level only
- tracks: `VenusX_Res_Act_*` and `VenusX_Res_BindI_*`
- experiments: `E0`, `E1`, `E2`, `E3`
- models: `mock`, `replay`, `openrouter`
- output: up to 3 ranked candidate labels, with `top_ids[0]` treated as the final prediction and a short `reasoning_summary`

## 1. Goal of the First Round

The first round should answer three questions:

1. Can general LLMs do the fragment-level task at all under the current prompt format?
2. Which prompt setting is best among `E1`, `E2`, and `E3`?
3. How do a few strong but relatively cheap baseline models compare on active-site and binding-site tracks?

The first round should not try to do everything at once.

Keep it focused on:

- clean reproducible runs
- a small prompt ablation
- a small but credible baseline model set
- paper-style reporting with LLM-specific diagnostics kept separate

## 2. Current Dataset Scope

Supported datasets:

- `VenusX_Res_Act_MF50`
- `VenusX_Res_Act_MF70`
- `VenusX_Res_Act_MF90`
- `VenusX_Res_BindI_MF50`
- `VenusX_Res_BindI_MF70`
- `VenusX_Res_BindI_MF90`

Current split sizes:

- `VenusX_Res_Act_MF50`: train `1488`, valid `186`, test `186`
- `VenusX_Res_Act_MF70`: train `2724`, valid `340`, test `341`
- `VenusX_Res_Act_MF90`: train `5269`, valid `659`, test `659`
- `VenusX_Res_BindI_MF50`: train `1640`, valid `205`, test `205`
- `VenusX_Res_BindI_MF70`: train `3016`, valid `377`, test `377`
- `VenusX_Res_BindI_MF90`: train `5306`, valid `663`, test `664`

Label-space size:

- active-site: `132` labels overall
- binding-site: `76` labels overall

This matters for cost:

- active-site prompts are longer because the full label catalog is larger
- `E3` is more expensive than `E2` because it includes the full sequence context

## 3. Metrics To Report

### Main paper table

These are the headline metrics:

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `mcc`

These should be used in the main result tables in the paper.

### Supplemental LLM table

These diagnose LLM behavior rather than pure classification quality:

- `top3_acc`
- `parse_success_rate`
- `invalid_label_rate`
- `abstain_rate`
- `coverage`
- `selective_accuracy`

These should be shown in an appendix, supplement, or ablation table.

### Recommended reporting rule

Malformed outputs, invalid labels, and backend failures should remain counted as wrong in the main paper table.

That keeps the benchmark honest and avoids hiding prompt-following failures.

## 4. Readiness Review

After reviewing `evaluation_llm/`, the framework is ready for a first experiment round.

What is in good shape:

- dataset loading is simple and deterministic
- label catalog loading is clean and stable
- prompt construction is explicit and easy to inspect
- parser is reasonably robust
- metrics now separate paper-style results from LLM diagnostics
- artifacts are detailed enough for error analysis
- backend failures are recorded per example instead of aborting the whole run

What to keep in mind:

- there is no retry or resume logic for OpenRouter requests
- cost and latency are not yet summarized into metrics, though OpenRouter usage is stored in per-example metadata when available
- `E3` can become noticeably more expensive than `E2`
- the framework currently uses the full candidate catalog in every prompt, so prompt length grows with label count

Bottom line:

- yes, it is ready for a first baseline round
- no, I would not start with the largest model set on the largest splits immediately

## 5. Recommended Model Groups

### Minimal first table

Start with 4 models:

- `deepseek/deepseek-chat-v3.1`
- `openai/gpt-5-mini`
- `google/gemini-2.5-flash`
- `anthropic/claude-haiku-4.5`

Why this group:

- 3 strong closed-model workhorses
- 1 strong open-family baseline
- reasonable cost for a first comparison

### Full starter set

If the minimal first table looks good, add:

- `openai/gpt-5`

## 6. Experiment Order

Run experiments from simple to hard.

### Stage 0: Local smoke test

Goal:

- verify the pipeline, artifacts, and local environment

Commands:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E0

python -m unittest tests.test_evaluation_llm
```

Success criteria:

- run completes
- `metrics.json` is created
- `records.jsonl` and `errors.jsonl` are created
- unit tests pass

### Stage 1: Prompt sanity check on a tiny slice

Goal:

- inspect real prompts before spending meaningful API budget
- confirm that the label cards and output format look reasonable

Recommended runs:

- `E1` on `10` examples
- `E2` on `10` examples
- `E3` on `10` examples

Use one very cheap model first:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_BindI_MF50 \
  --split test \
  --experiment E1 \
  --model_provider openrouter \
  --model_name deepseek/deepseek-chat-v3.1 \
  --max_examples 10
```

Repeat for `E2` and `E3`.

What to inspect:

- prompts in `records.jsonl`
- parse success in `metrics.json`
- whether the model returns 1 to 5 `top_ids`
- whether invalid labels appear

Stop if:

- parse success is poor
- invalid-label rate is high
- `E3` prompt size looks too expensive for the intended model

### Stage 2: Small validation ablation

Goal:

- choose between `E1`, `E2`, and `E3` on a real but still limited slice

Recommended setup:

- dataset: `MF50`
- split: `test`
- models: `deepseek/deepseek-chat-v3.1` and `openai/gpt-5-mini`
- cap: `50` to `100` examples

Suggested commands:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --split test \
  --experiment E2 \
  --model_provider openrouter \
  --model_name deepseek/deepseek-chat-v3.1 \
  --max_examples 100
```

Run `E1`, `E2`, and `E3` for both models.

Decision rule:

- prefer the best `main_paper_table.accuracy`
- break ties with `macro_f1`, then `mcc`
- only keep `E3` if it gives a clear improvement worth the extra cost

Practical expectation:

- `E2` is the strongest default candidate
- `E1` is the cheapest
- `E3` is the “maybe helpful, maybe overpriced” condition

### Stage 3: Full prompt selection on `MF50`

Goal:

- choose a frozen prompt configuration per track

Recommended setup:

- dataset: `VenusX_Res_Act_MF50` and `VenusX_Res_BindI_MF50`
- split: `valid`
- models: `deepseek/deepseek-chat-v3.1` and `openai/gpt-5-mini`
- no `max_examples`

Simplest route:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E2 \
  --model_provider openrouter \
  --model_name deepseek/deepseek-chat-v3.1 \
  --suite
```

Important note:

- `--suite` selects among `E1`, `E2`, and `E3` using the validation split and then evaluates the chosen configuration on test sets
- if you want strictly validation-only prompt selection first, run `E1` to `E3` manually on `--split valid`

### Stage 4: First full paper-style benchmark on `MF50`

Goal:

- build the first real comparison table

Recommended setup:

- datasets:
  - `VenusX_Res_Act_MF50`
  - `VenusX_Res_BindI_MF50`
- models:
  - `deepseek/deepseek-chat-v3.1`
  - `openai/gpt-5-mini`
  - `google/gemini-2.5-flash`
  - `anthropic/claude-haiku-4.5`
- prompt setting:
  - use the best setting chosen in Stage 3

Recommended reporting:

- one main paper table for each track
- one supplemental LLM table for each track

### Stage 5: Generalization to `MF70` and `MF90`

Goal:

- test whether the chosen setup transfers to less strict and more permissive similarity regimes without retuning

Recommended setup:

- freeze the prompt setting chosen on `MF50`
- run the same setting on:
  - `MF70`
  - `MF90`

Command pattern:

```bash
python -m evaluation_llm \
  --dataset_id VenusX_Res_Act_MF70 \
  --experiment E2 \
  --model_provider openrouter \
  --model_name deepseek/deepseek-chat-v3.1
```

Use the frozen experiment setting from Stage 3, even if it is `E1` or `E3` instead of `E2`.

### Stage 6: Full starter-model sweep

Goal:

- expand the comparison table after the minimal first table is stable

Command:

```bash
python -m evaluation_llm.run_openrouter_model_set \
  --dataset_id VenusX_Res_Act_MF50 \
  --experiment E2 \
  --model_set starter
```

Do this only after the manual smaller runs look healthy.

Reason:

- model-set runs are convenient
- but they are also an easy way to spend more money than expected

### Stage 7: Harder follow-up experiments

Only do these after the baseline round is complete:

- compare `E2` vs `E3` on both tracks at full scale
- add explicit cost and latency summaries
- add agent-style evaluation
- add retrieval-assisted or tool-using settings later

## 7. Recommended First-Round Sequence

If you want one concrete path to follow, use this:

1. `E0` smoke test locally.
2. Tiny `10`-example OpenRouter runs for `E1`, `E2`, `E3` on `VenusX_Res_BindI_MF50`.
3. `50`- to `100`-example validation ablation for:
   - `deepseek/deepseek-chat-v3.1`
   - `openai/gpt-5-mini`
4. Choose one prompt setting per track.
5. Run full `MF50` tests for the 4-model minimal table.
6. Freeze the setting and transfer to `MF70` and `MF90`.
7. Add `openai/gpt-5` if budget allows.

This sequence is conservative, cheap enough for a first round, and strong enough to produce an initial paper-quality result table.

## 8. What To Save and Compare

For every run, keep:

- `resolved_config.json`
- `metrics.json`
- `records.jsonl`
- `errors.jsonl`

During comparison, pay attention to:

- `main_paper_table.accuracy`
- `main_paper_table.macro_f1`
- `main_paper_table.mcc`
- `supplemental_llm_table.parse_success_rate`
- `supplemental_llm_table.invalid_label_rate`
- `supplemental_llm_table.coverage`

This helps separate:

- true classification performance
- prompt-following quality
- provider/runtime stability

## 9. Practical Advice

- Start with binding-site before active-site if you want a cheaper prompt regime. The binding-site catalog is smaller.
- Use `E2` as the default unless validation clearly favors `E1` or `E3`.
- Keep `temperature=0.0` for the first benchmark round.
- Do not start with `MF90` full sweeps. It is larger and easier to overspend on.
- Read a few `records.jsonl` examples before trusting the aggregate metrics.
- Treat `errors.jsonl` as part of the benchmark, not just debugging output.

## 10. Exit Criteria For The First Round

You are done with the first round when you have:

- one chosen prompt setting per track
- a 4-model minimal comparison table on `MF50`
- frozen-transfer results on `MF70` and `MF90`
- both the main paper table and supplemental LLM table
- a short qualitative review of common failure cases from `records.jsonl`

That is enough to support a first real discussion of whether fragment-level LLM benchmarking is working and where to go next.
