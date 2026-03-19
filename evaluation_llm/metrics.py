from __future__ import annotations

from evaluation_llm.fragment_dataset import fragment_length_bin
from evaluation_llm.records import ExampleResult


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


class FragmentBenchmarkMetrics:
    def __init__(self) -> None:
        self.results: list[ExampleResult] = []

    def update(self, result: ExampleResult) -> None:
        self.results.append(result)

    def compute(self) -> dict:
        overall = self._summarize(self.results)
        slices = {
            "seen_in_train": self._summarize([result for result in self.results if result.seen_in_train]),
            "unseen_in_train": self._summarize([result for result in self.results if not result.seen_in_train]),
            "multi_fragment": self._summarize([result for result in self.results if result.example.is_multi_fragment]),
            "single_fragment": self._summarize([result for result in self.results if not result.example.is_multi_fragment]),
        }

        for length_bucket in ("short", "medium", "long"):
            slices[f"fragment_length::{length_bucket}"] = self._summarize(
                [
                    result
                    for result in self.results
                    if fragment_length_bin(result.example.fragment_length) == length_bucket
                ]
            )

        return {
            "overall": overall,
            "slices": slices,
        }

    def _summarize(self, results: list[ExampleResult]) -> dict:
        count = len(results)
        if count == 0:
            return {"count": 0}

        gold_labels = [result.example.interpro_id for result in results]
        predicted_labels = [result.predicted_top_id or "__NONE__" for result in results]
        per_class_f1, macro_f1 = _compute_label_f1(gold_labels, predicted_labels)

        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        parse_success = 0
        invalid_label_examples = 0
        abstain_examples = 0

        for result in results:
            predictions = list(result.prediction.top_ids)
            if result.prediction.parse_success:
                parse_success += 1
            if result.prediction.invalid_labels:
                invalid_label_examples += 1
            if result.prediction.abstain:
                abstain_examples += 1
            if predictions[:1] == [result.example.interpro_id]:
                top1_hits += 1
            if result.example.interpro_id in predictions[:3]:
                top3_hits += 1
            if result.example.interpro_id in predictions[:5]:
                top5_hits += 1

        return {
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
