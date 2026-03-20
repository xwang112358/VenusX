from __future__ import annotations

import math

from evaluation_llm.records import ExampleResult

MISSING_PREDICTION_LABEL = "__NONE__"


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


def _round_metric(value: float) -> float:
    return round(value, 6)


def _compute_macro_classification_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
) -> tuple[float, float, float]:
    if not gold_labels:
        return 0.0, 0.0, 0.0

    labels = sorted(set(gold_labels) | {label for label in predicted_labels if label != MISSING_PREDICTION_LABEL})
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = _f1(tp, fp, fn)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    macro_precision = _round_metric(sum(precision_values) / len(precision_values))
    macro_recall = _round_metric(sum(recall_values) / len(recall_values))
    macro_f1 = _round_metric(sum(f1_values) / len(f1_values))
    return macro_precision, macro_recall, macro_f1


def _compute_multiclass_mcc(gold_labels: list[str], predicted_labels: list[str]) -> float:
    labels = sorted(set(gold_labels) | set(predicted_labels))
    if not labels:
        return 0.0

    index_by_label = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for gold, pred in zip(gold_labels, predicted_labels):
        matrix[index_by_label[gold]][index_by_label[pred]] += 1

    true_totals = [sum(row) for row in matrix]
    pred_totals = [sum(matrix[row_index][col_index] for row_index in range(len(labels))) for col_index in range(len(labels))]
    total = sum(true_totals)
    correct = sum(matrix[index][index] for index in range(len(labels)))
    numerator = correct * total - sum(
        pred_total * true_total for pred_total, true_total in zip(pred_totals, true_totals)
    )
    denominator_left = total * total - sum(pred_total * pred_total for pred_total in pred_totals)
    denominator_right = total * total - sum(true_total * true_total for true_total in true_totals)
    denominator = math.sqrt(max(denominator_left, 0) * max(denominator_right, 0))
    if denominator == 0.0:
        return 0.0
    return _round_metric(numerator / denominator)


class FragmentBenchmarkMetrics:
    def __init__(self) -> None:
        self.results: list[ExampleResult] = []

    def update(self, result: ExampleResult) -> None:
        self.results.append(result)

    def compute(self) -> dict:
        return self._summarize(self.results)

    def _summarize(self, results: list[ExampleResult]) -> dict:
        count = len(results)
        if count == 0:
            return {
                "main_paper_table": {"count": 0},
                "supplemental_llm_table": {"count": 0},
            }

        gold_labels = [result.example.interpro_id for result in results]
        predicted_labels = [result.predicted_top_id or MISSING_PREDICTION_LABEL for result in results]
        macro_precision, macro_recall, macro_f1 = _compute_macro_classification_metrics(
            gold_labels, predicted_labels
        )
        mcc = _compute_multiclass_mcc(gold_labels, predicted_labels)

        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        parse_success = 0
        invalid_label_examples = 0
        abstain_examples = 0
        covered_examples = 0

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
            if predictions[:1]:
                covered_examples += 1
            if result.example.interpro_id in predictions[:3]:
                top3_hits += 1
            if result.example.interpro_id in predictions[:5]:
                top5_hits += 1

        main_paper_table = {
            "count": count,
            "accuracy": _safe_rate(top1_hits, count),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "mcc": mcc,
        }
        supplemental_llm_table = {
            "count": count,
            "top3_acc": _safe_rate(top3_hits, count),
            "top5_acc": _safe_rate(top5_hits, count),
            "parse_success_rate": _safe_rate(parse_success, count),
            "invalid_label_rate": _safe_rate(invalid_label_examples, count),
            "abstain_rate": _safe_rate(abstain_examples, count),
            "coverage": _safe_rate(covered_examples, count),
            "selective_accuracy": _safe_rate(top1_hits, covered_examples),
        }
        return {
            "main_paper_table": main_paper_table,
            "supplemental_llm_table": supplemental_llm_table,
        }
