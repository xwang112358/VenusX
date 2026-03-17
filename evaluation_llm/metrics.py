from __future__ import annotations

from collections import defaultdict

from evaluation_llm.datasets import fragment_length_bin
from evaluation_llm.interfaces import MetricSuite
from evaluation_llm.types import EvaluationRecord


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_label_f1(gold_labels: list[str], predicted_labels: list[str]) -> tuple[dict[str, float], float]:
    if not gold_labels:
        return {}, 0.0

    labels = sorted(set(gold_labels))
    per_class = {}
    for label in labels:
        tp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == label and pred != label)
        per_class[label] = round(_f1(tp, fp, fn), 6)
    macro = round(sum(per_class.values()) / len(per_class), 6)
    return per_class, macro


class FragmentMetricSuite(MetricSuite):
    def __init__(self) -> None:
        self.records: list[EvaluationRecord] = []

    def update(self, record: EvaluationRecord) -> None:
        self.records.append(record)

    def compute(self) -> dict:
        overall = self._summarize(self.records)
        slices = {}
        slices["seen_in_train"] = self._summarize([record for record in self.records if record.seen_in_train])
        slices["unseen_in_train"] = self._summarize([record for record in self.records if not record.seen_in_train])
        slices["multi_fragment"] = self._summarize([record for record in self.records if record.example.is_multi_fragment])
        slices["single_fragment"] = self._summarize([record for record in self.records if not record.example.is_multi_fragment])

        for length_bucket in ("short", "medium", "long"):
            slices[f"fragment_length::{length_bucket}"] = self._summarize(
                [
                    record
                    for record in self.records
                    if fragment_length_bin(record.example.fragment_length) == length_bucket
                ]
            )

        return {
            "overall": overall,
            "slices": slices,
        }

    def _summarize(self, records: list[EvaluationRecord]) -> dict:
        count = len(records)
        if count == 0:
            return {"count": 0}

        gold_labels = [record.example.interpro_id for record in records]
        predicted_labels = [record.predicted_top_id or "__NONE__" for record in records]
        per_class_f1, macro_f1 = _compute_label_f1(gold_labels, predicted_labels)

        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        parse_success = 0
        invalid_label_examples = 0
        abstain_examples = 0
        candidate_hits = 0
        oracle_hits = 0
        in_candidate_predictions = 0
        retrieval_count = 0

        for record in records:
            predictions = list(record.parsed.top_ids)
            if record.parsed.parse_success:
                parse_success += 1
            if record.parsed.invalid_labels:
                invalid_label_examples += 1
            if record.parsed.abstain:
                abstain_examples += 1
            if predictions[:1] == [record.example.interpro_id]:
                top1_hits += 1
            if record.example.interpro_id in predictions[:3]:
                top3_hits += 1
            if record.example.interpro_id in predictions[:5]:
                top5_hits += 1

            if record.candidate_hit is not None:
                retrieval_count += 1
                if record.candidate_hit:
                    candidate_hits += 1
                    oracle_hits += 1
            if record.prediction_in_candidates is not None and record.prediction_in_candidates:
                in_candidate_predictions += 1

        summary = {
            "count": count,
            "top1_acc": _safe_rate(top1_hits, count),
            "top3_acc": _safe_rate(top3_hits, count),
            "top5_acc": _safe_rate(top5_hits, count),
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "parse_success_rate": _safe_rate(parse_success, count),
            "invalid_label_rate": _safe_rate(invalid_label_examples, count),
            "abstain_rate": _safe_rate(abstain_examples, count),
        }

        if retrieval_count > 0:
            summary["candidate_recall@K"] = _safe_rate(candidate_hits, retrieval_count)
            summary["oracle_top1@K"] = _safe_rate(oracle_hits, retrieval_count)
            summary["prediction_in_candidates_rate"] = _safe_rate(in_candidate_predictions, retrieval_count)

        return summary
