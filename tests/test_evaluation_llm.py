from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation_llm.fragment_dataset import (
    get_dataset_info,
    load_fragment_examples,
    summarize_catalog_alignment,
)
from evaluation_llm.label_catalog import load_label_catalog
from evaluation_llm.metrics import FragmentBenchmarkMetrics
from evaluation_llm.model_backends import create_model_backend
from evaluation_llm.model_sets import get_model_set, list_model_sets
from evaluation_llm.prompt_and_parse import build_fragment_prompt, parse_model_response
from evaluation_llm.records import ExampleResult, ExperimentSettings, FragmentExample, Prediction
from evaluation_llm.run_fragment_benchmark import build_suite_settings, run_single_benchmark


def make_example(
    uid: str,
    interpro_id: str,
    interpro_label: int,
    seq_fragment_raw: str,
    fragment_parts: tuple[str, ...],
) -> FragmentExample:
    starts = []
    ends = []
    offset = 1
    for fragment in fragment_parts:
        starts.append(offset)
        ends.append(offset + len(fragment) - 1)
        offset = ends[-1] + 1

    return FragmentExample(
        uid=uid,
        dataset_id="VenusX_Res_Act_MF50",
        split="test",
        interpro_id=interpro_id,
        interpro_label=interpro_label,
        seq_fragment_raw=seq_fragment_raw,
        fragment_parts=fragment_parts,
        seq_full="".join(fragment_parts),
        start_parts=tuple(starts),
        end_parts=tuple(ends),
        is_multi_fragment=len(fragment_parts) > 1,
    )


class FragmentDatasetTests(unittest.TestCase):
    def test_multi_fragment_csv_parsing_preserves_alignment(self) -> None:
        dataset_info = get_dataset_info("VenusX_Res_BindI_MF50")
        examples = load_fragment_examples(dataset_info, split="train")
        multi_fragment = next(example for example in examples if example.is_multi_fragment)
        self.assertEqual(multi_fragment.seq_fragment_raw.count("|") + 1, len(multi_fragment.fragment_parts))
        self.assertEqual(len(multi_fragment.fragment_parts), len(multi_fragment.start_parts))
        self.assertEqual(len(multi_fragment.fragment_parts), len(multi_fragment.end_parts))
        self.assertTrue(all(start <= end for start, end in multi_fragment.ranges()))

    def test_catalog_alignment_for_active_and_binding(self) -> None:
        active_info = get_dataset_info("VenusX_Res_Act_MF50")
        active_catalog = load_label_catalog(active_info.catalog_path)
        active_examples = load_fragment_examples(active_info, split="test")
        active_summary = summarize_catalog_alignment(active_examples, active_catalog)
        self.assertGreater(active_summary["catalog_index_mismatch_count"], 0)

        binding_info = get_dataset_info("VenusX_Res_BindI_MF50")
        binding_catalog = load_label_catalog(binding_info.catalog_path)
        binding_examples = load_fragment_examples(binding_info, split="test")
        binding_summary = summarize_catalog_alignment(binding_examples, binding_catalog)
        self.assertEqual(binding_summary["catalog_index_mismatch_count"], 0)


class PromptAndParsingTests(unittest.TestCase):
    def test_prompt_is_deterministic_and_parser_normalizes_outputs(self) -> None:
        dataset_info = get_dataset_info("VenusX_Res_Act_MF50")
        catalog = load_label_catalog(dataset_info.catalog_path)
        example = load_fragment_examples(dataset_info, split="test", max_examples=1)[0]
        settings = ExperimentSettings(
            dataset_id=dataset_info.dataset_id,
            label_card_style="short_desc",
            include_full_sequence=True,
        )

        prompt_a = build_fragment_prompt(example, catalog, settings)
        prompt_b = build_fragment_prompt(example, catalog, settings)
        self.assertEqual(prompt_a, prompt_b)
        self.assertIn("at most 5 candidates", prompt_a)
        self.assertIn("return between 1 and 5 candidate accessions", prompt_a)

        exact = parse_model_response(
            f'{{"top_ids":["{example.interpro_id}"],"confidence":0.9,"abstain":false}}',
            catalog,
        )
        self.assertEqual(exact.top_ids, (example.interpro_id,))
        self.assertTrue(exact.parse_success)

        name_only = parse_model_response(
            '{"top_ids":["Hydroxymethylglutaryl-CoA lyase, active site"],"confidence":0.6,"abstain":false}',
            catalog,
        )
        self.assertEqual(name_only.top_ids, ("IPR000138",))

        sample_ids = [card.accession for card in catalog.sorted_cards()[:6]]
        max_five = parse_model_response(
            '{"top_ids":'
            + str(sample_ids).replace("'", '"')
            + ',"confidence":0.7,"abstain":false}',
            catalog,
        )
        self.assertEqual(len(max_five.top_ids), 5)
        self.assertEqual(max_five.top_ids, tuple(sample_ids[:5]))

        invalid = parse_model_response("not json and no accession", catalog)
        self.assertFalse(invalid.parse_success)

        abstain = parse_model_response('{"top_ids":[],"confidence":0.2,"abstain":true}', catalog)
        self.assertTrue(abstain.parse_success)
        self.assertTrue(abstain.abstain)
        self.assertEqual(abstain.top_ids, ())


class MetricTests(unittest.TestCase):
    def test_metrics_handle_hits_misses_and_parse_failures(self) -> None:
        metrics = FragmentBenchmarkMetrics()
        example_a = make_example("a", "IPR000126", 0, "AAAA", ("AAAA",))
        example_b = make_example("b", "IPR000138", 1, "BBBB", ("BBBB",))
        example_c = make_example("c", "IPR017950", 35, "CCCC|DDDD", ("CCCC", "DDDD"))

        results = [
            ExampleResult(
                example=example_a,
                prompt="prompt",
                raw_response='{"top_ids":["IPR000126"]}',
                response_metadata={},
                prediction=Prediction(("IPR000126",), 1.0, False, True, ()),
                predicted_top_id="IPR000126",
            ),
            ExampleResult(
                example=example_b,
                prompt="prompt",
                raw_response='{"top_ids":["IPR000126","IPR000138"]}',
                response_metadata={},
                prediction=Prediction(("IPR000126", "IPR000138"), 0.9, False, True, ()),
                predicted_top_id="IPR000126",
            ),
            ExampleResult(
                example=example_c,
                prompt="prompt",
                raw_response="bad output",
                response_metadata={},
                prediction=Prediction((), None, False, False, (), "No valid JSON"),
                predicted_top_id=None,
            ),
        ]

        for result in results:
            metrics.update(result)

        summary = metrics.compute()
        self.assertEqual(summary["main_paper_table"]["accuracy"], 0.333333)
        self.assertEqual(summary["main_paper_table"]["macro_precision"], 0.166667)
        self.assertEqual(summary["main_paper_table"]["macro_recall"], 0.333333)
        self.assertEqual(summary["main_paper_table"]["macro_f1"], 0.222222)
        self.assertEqual(summary["main_paper_table"]["mcc"], 0.204124)
        self.assertEqual(summary["supplemental_llm_table"]["top3_acc"], 0.666667)
        self.assertEqual(summary["supplemental_llm_table"]["parse_success_rate"], 0.666667)
        self.assertEqual(summary["supplemental_llm_table"]["coverage"], 0.666667)
        self.assertEqual(summary["supplemental_llm_table"]["selective_accuracy"], 0.5)


class RunnerPresetTests(unittest.TestCase):
    def test_suite_only_uses_e1_to_e3(self) -> None:
        base_settings = ExperimentSettings(dataset_id="VenusX_Res_Act_MF50")
        settings_list = build_suite_settings(base_settings)
        self.assertEqual(
            [settings.experiment_name for settings in settings_list],
            [
                "E1_FullCatalogName",
                "E2_FullCatalogShortDesc",
                "E3_FullCatalogContext",
            ],
        )

    def test_openrouter_model_sets_include_expected_starter_models(self) -> None:
        self.assertEqual(list_model_sets(), ["extended", "starter"])

        starter_ids = [spec.model_id for spec in get_model_set("starter")]
        self.assertEqual(
            starter_ids,
            [
                "google/gemini-2.5-flash-lite",
                "openai/gpt-4.1-mini",
                "deepseek/deepseek-chat-v3.1",
                "meta-llama/llama-3.3-70b-instruct",
                "qwen/qwen-2.5-72b-instruct",
            ],
        )

        extended_ids = [spec.model_id for spec in get_model_set("extended")]
        self.assertIn("anthropic/claude-3.5-haiku", extended_ids)
        self.assertIn("mistralai/mistral-small-3.2-24b-instruct", extended_ids)


class OpenRouterBackendTests(unittest.TestCase):
    def test_openrouter_backend_builds_expected_request_from_env_file(self) -> None:
        dataset_info = get_dataset_info("VenusX_Res_Act_MF50")
        catalog = load_label_catalog(dataset_info.catalog_path)
        example = load_fragment_examples(dataset_info, split="test", max_examples=1)[0]
        label_cards = catalog.sorted_cards()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "gen-123",
            "model": "openai/gpt-4.1-mini",
            "usage": {"prompt_tokens": 123, "completion_tokens": 17},
            "choices": [
                {
                    "message": {
                        "content": '{"top_ids":["IPR000138"],"confidence":0.8,"abstain":false}'
                    }
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                'OPENROUTER_API_KEY="test-key"\n'
                'OPENROUTER_HTTP_REFERER="https://example.com/venusx"\n'
                'OPENROUTER_TITLE="VenusX"\n'
            )

            with patch.dict(
                os.environ,
                {
                    "OPENROUTER_ENV_FILE": str(env_path),
                },
                clear=False,
            ):
                os.environ.pop("OPENROUTER_API_KEY", None)
                os.environ.pop("OPENROUTER_HTTP_REFERER", None)
                os.environ.pop("OPENROUTER_TITLE", None)
                with patch("evaluation_llm.model_backends.requests.post", return_value=mock_response) as mock_post:
                    backend = create_model_backend(
                        model_provider="openrouter",
                        model_name="openai/gpt-4.1-mini",
                        catalog=catalog,
                        temperature=0.2,
                    )
                    response = backend.generate(prompt="test prompt", example=example, label_cards=label_cards)

        self.assertIn("IPR000138", response.raw_text)
        self.assertEqual(response.metadata["mode"], "openrouter")
        self.assertEqual(response.metadata["response_model"], "openai/gpt-4.1-mini")

        self.assertEqual(mock_post.call_count, 1)
        call_args = mock_post.call_args
        self.assertEqual(call_args.args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(call_args.kwargs["headers"]["HTTP-Referer"], "https://example.com/venusx")
        self.assertEqual(call_args.kwargs["headers"]["X-OpenRouter-Title"], "VenusX")
        self.assertEqual(call_args.kwargs["json"]["model"], "openai/gpt-4.1-mini")
        self.assertEqual(call_args.kwargs["json"]["messages"][0]["content"], "test prompt")
        self.assertEqual(call_args.kwargs["json"]["temperature"], 0.2)
        self.assertEqual(call_args.kwargs["json"]["user"], example.uid)


class EndToEndSmokeTests(unittest.TestCase):
    def test_run_single_smoke_writes_artifacts(self) -> None:
        settings = ExperimentSettings(
            dataset_id="VenusX_Res_Act_MF50",
            split="test",
            experiment_name="smoke_test",
            label_card_style="name_only",
            model_provider="mock",
            model_name="oracle",
            max_examples=5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics, run_dir = run_single_benchmark(settings, artifact_root=Path(tmpdir))
            self.assertEqual(metrics["main_paper_table"]["accuracy"], 1.0)
            self.assertTrue((run_dir / "resolved_config.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "records.jsonl").exists())
            self.assertTrue((run_dir / "errors.jsonl").exists())

    def test_run_single_continues_when_backend_fails_for_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "missing.jsonl"
            replay_path.write_text('{"uid":"does-not-match","raw_text":"{\\"top_ids\\":[\\"IPR000138\\"],\\"abstain\\":false}"}\n')
            settings = ExperimentSettings(
                dataset_id="VenusX_Res_Act_MF50",
                split="test",
                experiment_name="backend_error_smoke",
                label_card_style="name_only",
                model_provider="replay",
                model_name=str(replay_path),
                max_examples=1,
            )

            metrics, run_dir = run_single_benchmark(settings, artifact_root=Path(tmpdir))
            self.assertEqual(metrics["main_paper_table"]["count"], 1)
            self.assertEqual(metrics["main_paper_table"]["accuracy"], 0.0)
            self.assertEqual(metrics["supplemental_llm_table"]["parse_success_rate"], 0.0)
            self.assertTrue((run_dir / "errors.jsonl").read_text().strip())


if __name__ == "__main__":
    unittest.main()
