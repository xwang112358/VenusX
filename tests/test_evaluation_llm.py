from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation_llm.catalog import load_label_catalog
from evaluation_llm.datasets import inspect_catalog_alignment, load_fragment_examples
from evaluation_llm.metrics import FragmentMetricSuite
from evaluation_llm.parsing import JsonResponseParser
from evaluation_llm.prompting import FragmentPromptBuilder
from evaluation_llm.registry import get_dataset_spec
from evaluation_llm.retrieval import TopKPrototypeCandidateProvider
from evaluation_llm.runner import run_single
from evaluation_llm.types import (
    CandidateRecord,
    EvaluationRecord,
    FragmentExample,
    ParsedPrediction,
    PromptContext,
    RunConfig,
)


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
        spec = get_dataset_spec("VenusX_Res_BindI_MF50")
        examples = load_fragment_examples(spec, split="train")
        multi_fragment = next(example for example in examples if example.is_multi_fragment)
        self.assertEqual(multi_fragment.seq_fragment_raw.count("|") + 1, len(multi_fragment.fragment_parts))
        self.assertEqual(len(multi_fragment.fragment_parts), len(multi_fragment.start_parts))
        self.assertEqual(len(multi_fragment.fragment_parts), len(multi_fragment.end_parts))
        self.assertTrue(all(start <= end for start, end in multi_fragment.ranges()))

    def test_catalog_alignment_for_active_and_binding(self) -> None:
        active_spec = get_dataset_spec("VenusX_Res_Act_MF50")
        active_catalog = load_label_catalog(active_spec.catalog_path)
        active_examples = load_fragment_examples(active_spec, split="test")
        active_summary = inspect_catalog_alignment(active_examples, active_catalog)
        self.assertGreater(active_summary["catalog_index_mismatch_count"], 0)

        binding_spec = get_dataset_spec("VenusX_Res_BindI_MF50")
        binding_catalog = load_label_catalog(binding_spec.catalog_path)
        binding_examples = load_fragment_examples(binding_spec, split="test")
        binding_summary = inspect_catalog_alignment(binding_examples, binding_catalog)
        self.assertEqual(binding_summary["catalog_index_mismatch_count"], 0)


class PromptAndParsingTests(unittest.TestCase):
    def test_prompt_is_deterministic_and_parser_normalizes_outputs(self) -> None:
        spec = get_dataset_spec("VenusX_Res_Act_MF50")
        catalog = load_label_catalog(spec.catalog_path)
        example = load_fragment_examples(spec, split="test", max_examples=1)[0]
        candidate_accessions = sorted(
            [example.interpro_id, "IPR000126", "IPR000138"],
        )
        candidate_cards = catalog.cards_for_accessions(candidate_accessions)
        candidates = [
            CandidateRecord(accession=card.accession, score=1.0, rank=index + 1, source="test")
            for index, card in enumerate(candidate_cards)
        ]
        context = PromptContext(
            example=example,
            config=RunConfig(
                dataset_id=spec.dataset_id,
                label_card_style="short_desc",
                include_full_sequence=True,
            ),
            candidate_records=tuple(candidates),
            candidate_cards=candidate_cards,
        )

        builder = FragmentPromptBuilder()
        prompt_a = builder.build(context, catalog).prompt
        prompt_b = builder.build(context, catalog).prompt
        self.assertEqual(prompt_a, prompt_b)

        parser = JsonResponseParser()
        exact = parser.parse(f'{{"top_ids":["{example.interpro_id}"],"confidence":0.9,"abstain":false}}', catalog)
        self.assertEqual(exact.top_ids, (example.interpro_id,))
        self.assertTrue(exact.parse_success)

        name_only = parser.parse(
            '{"top_ids":["Hydroxymethylglutaryl-CoA lyase, active site"],"confidence":0.6,"abstain":false}',
            catalog,
        )
        self.assertEqual(name_only.top_ids, ("IPR000138",))

        invalid = parser.parse("not json and no accession", catalog)
        self.assertFalse(invalid.parse_success)

        abstain = parser.parse('{"top_ids":[],"confidence":0.2,"abstain":true}', catalog)
        self.assertTrue(abstain.parse_success)
        self.assertTrue(abstain.abstain)
        self.assertEqual(abstain.top_ids, ())


class MetricSuiteTests(unittest.TestCase):
    def test_metric_suite_handles_hits_misses_and_parse_failures(self) -> None:
        metrics = FragmentMetricSuite()
        example_a = make_example("a", "IPR000126", 0, "AAAA", ("AAAA",))
        example_b = make_example("b", "IPR000138", 1, "BBBB", ("BBBB",))
        example_c = make_example("c", "IPR017950", 35, "CCCC|DDDD", ("CCCC", "DDDD"))

        records = [
            EvaluationRecord(
                example=example_a,
                candidates=(CandidateRecord("IPR000126", 1.0, 1, "test"),),
                parsed=ParsedPrediction(("IPR000126",), 1.0, False, True, ()),
                prompt="prompt",
                raw_response='{"top_ids":["IPR000126"]}',
                response_metadata={},
                seen_in_train=True,
                predicted_top_id="IPR000126",
                candidate_hit=True,
                prediction_in_candidates=True,
            ),
            EvaluationRecord(
                example=example_b,
                candidates=(
                    CandidateRecord("IPR000126", 0.8, 1, "test"),
                    CandidateRecord("IPR000138", 0.7, 2, "test"),
                ),
                parsed=ParsedPrediction(("IPR000126", "IPR000138"), 0.9, False, True, ()),
                prompt="prompt",
                raw_response='{"top_ids":["IPR000126","IPR000138"]}',
                response_metadata={},
                seen_in_train=False,
                predicted_top_id="IPR000126",
                candidate_hit=True,
                prediction_in_candidates=True,
            ),
            EvaluationRecord(
                example=example_c,
                candidates=(CandidateRecord("IPR000126", 0.1, 1, "test"),),
                parsed=ParsedPrediction((), None, False, False, (), "No valid JSON"),
                prompt="prompt",
                raw_response="bad output",
                response_metadata={},
                seen_in_train=False,
                predicted_top_id=None,
                candidate_hit=False,
                prediction_in_candidates=False,
            ),
        ]

        for record in records:
            metrics.update(record)

        summary = metrics.compute()
        self.assertEqual(summary["overall"]["top1_acc"], 0.333333)
        self.assertEqual(summary["overall"]["top3_acc"], 0.666667)
        self.assertEqual(summary["overall"]["parse_success_rate"], 0.666667)
        self.assertEqual(summary["overall"]["candidate_recall@K"], 0.666667)


class RetrievalTests(unittest.TestCase):
    def test_retriever_uses_stable_tie_breaking(self) -> None:
        spec = get_dataset_spec("VenusX_Res_Act_MF50")
        catalog = load_label_catalog(spec.catalog_path)
        train_examples = [
            make_example("t1", "IPR000126", 0, "AAAAAA", ("AAAAAA",)),
            make_example("t2", "IPR000138", 1, "AAAAAA", ("AAAAAA",)),
        ]
        provider = TopKPrototypeCandidateProvider(catalog=catalog, train_examples=train_examples)
        query = make_example("q", "IPR000126", 0, "AAAAAA", ("AAAAAA",))
        candidates = provider.get_candidates(query, top_k=2)
        self.assertEqual([candidate.accession for candidate in candidates], ["IPR000126", "IPR000138"])


class EndToEndSmokeTests(unittest.TestCase):
    def test_run_single_smoke_writes_artifacts(self) -> None:
        config = RunConfig(
            dataset_id="VenusX_Res_Act_MF50",
            split="test",
            candidate_strategy="full_catalog",
            label_card_style="name_only",
            model_provider="mock",
            model_name="oracle",
            max_examples=5,
            experiment_name="smoke_test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics, run_dir = run_single(config, artifact_root=Path(tmpdir))
            self.assertEqual(metrics["overall"]["top1_acc"], 1.0)
            self.assertTrue((run_dir / "resolved_config.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "records.jsonl").exists())
            self.assertTrue((run_dir / "errors.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
