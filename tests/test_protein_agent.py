"""Unit tests for protein_agent/ and evaluation_agent/.

All HTTP calls and the OpenAI client are mocked — no network access required.
"""
from __future__ import annotations

import csv
import io
import json
import textwrap
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# protein_agent tests
# ---------------------------------------------------------------------------
from protein_agent.records import AgentResult, SiteAnnotation
from protein_agent.tools.interpro_scan import InterProScanTool, _parse_result_json

from evaluation_agent.dataset import load_examples, _parse_positions
from evaluation_agent.metrics import aggregate_results, fragment_metrics, residue_metrics
from evaluation_agent.records import EvalResult


# ── Fixtures ──────────────────────────────────────────────────────────────

ACTIVE_SITE_JSON: dict = {
    "results": [
        {
            "matches": [
                {
                    "signature": {
                        "entry": {
                            "accession": "IPR001270",
                            "name": "Serine protease, His active site",
                            "type": "ACTIVE_SITE",
                        }
                    },
                    "locations": [
                        {"start": 57, "end": 57},
                        {"start": 102, "end": 102},
                    ],
                }
            ]
        }
    ]
}

MULTI_MATCH_JSON: dict = {
    "results": [
        {
            "matches": [
                {
                    "signature": {
                        "entry": {
                            "accession": "IPR001270",
                            "name": "Serine protease, His active site",
                            "type": "ACTIVE_SITE",
                        }
                    },
                    "locations": [{"start": 57, "end": 57}],
                },
                {
                    "signature": {
                        "entry": {
                            "accession": "IPR001270",
                            "name": "Serine protease, His active site",
                            "type": "ACTIVE_SITE",
                        }
                    },
                    "locations": [{"start": 102, "end": 102}],
                },
            ]
        }
    ]
}

NON_IPR_JSON: dict = {
    "results": [
        {
            "matches": [
                {
                    "signature": {
                        "entry": {
                            "accession": "PS00142",  # not an IPR accession
                            "name": "PROSITE pattern",
                            "type": "ACTIVE_SITE",
                        }
                    },
                    "locations": [{"start": 1, "end": 10}],
                }
            ]
        }
    ]
}


# ── _parse_result_json ────────────────────────────────────────────────────


class ParseResultJsonTests(unittest.TestCase):
    def test_empty_input(self):
        result = _parse_result_json({})
        self.assertEqual(result, [])

    def test_empty_results_list(self):
        result = _parse_result_json({"results": []})
        self.assertEqual(result, [])

    def test_active_site_fixture(self):
        annotations = _parse_result_json(ACTIVE_SITE_JSON)
        self.assertEqual(len(annotations), 1)
        ann = annotations[0]
        self.assertEqual(ann.accession, "IPR001270")
        self.assertEqual(ann.site_type, "ACTIVE_SITE")
        self.assertIn((57, 57), ann.locations)
        self.assertIn((102, 102), ann.locations)

    def test_deduplication(self):
        """Same accession in two separate matches → single annotation, merged locations."""
        annotations = _parse_result_json(MULTI_MATCH_JSON)
        self.assertEqual(len(annotations), 1)
        ann = annotations[0]
        self.assertIn((57, 57), ann.locations)
        self.assertIn((102, 102), ann.locations)

    def test_non_ipr_accession_skipped(self):
        annotations = _parse_result_json(NON_IPR_JSON)
        self.assertEqual(annotations, [])


# ── InterProScanTool (HTTP mocked) ────────────────────────────────────────


class InterProScanToolTests(unittest.TestCase):
    @patch("protein_agent.tools.interpro_scan.requests.post")
    @patch("protein_agent.tools.interpro_scan.requests.get")
    def test_search_returns_annotations(self, mock_get, mock_post):
        # Simulate submit → status → result
        mock_post.return_value = MagicMock(status_code=200, text="jobId-123")

        status_response = MagicMock(status_code=200, text="FINISHED")
        result_response = MagicMock(status_code=200)
        result_response.json.return_value = ACTIVE_SITE_JSON
        mock_get.side_effect = [status_response, result_response]

        tool = InterProScanTool(email="test@example.com", poll_interval=0)
        annotations = tool.search("MKVLAAAA")

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].accession, "IPR001270")

    @patch("protein_agent.tools.interpro_scan.requests.post")
    def test_submit_failure_raises(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500, text="Server error")
        tool = InterProScanTool(email="test@example.com")
        from protein_agent.tools.interpro_scan import InterProScanError
        with self.assertRaises(InterProScanError):
            tool.search("MKVLAAAA")


# ── ProteinAgent tool-use loop (OpenAI client injected via _client) ────────


class ProteinAgentTests(unittest.TestCase):
    def _make_tool_call_response(self, tool_call_id: str, sequence: str):
        """Build a mock OpenAI response that requests a tool call."""
        tc = MagicMock()
        tc.id = tool_call_id
        tc.function.name = "search_interpro"
        tc.function.arguments = json.dumps({"sequence": sequence})

        msg = MagicMock()
        msg.tool_calls = [tc]

        choice = MagicMock()
        choice.finish_reason = "tool_calls"
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]
        response.usage.model_dump.return_value = {
            "prompt_tokens": 100, "completion_tokens": 50
        }
        return response

    def _make_stop_response(self):
        """Build a mock OpenAI response that ends the loop."""
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "Annotations retrieved."

        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]
        response.usage.model_dump.return_value = {
            "prompt_tokens": 200, "completion_tokens": 30
        }
        return response

    @patch("protein_agent.agent.InterProScanTool")
    def test_run_tool_loop(self, MockTool):
        annotation = SiteAnnotation(
            accession="IPR001270",
            name="Test",
            site_type="ACTIVE_SITE",
            locations=((57, 57),),
        )

        # Tool returns one annotation
        tool_instance = MagicMock()
        tool_instance.search.return_value = [annotation]
        MockTool.return_value = tool_instance

        # LLM: first calls tool, then ends
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            self._make_tool_call_response("tu-1", "MKVL"),
            self._make_stop_response(),
        ]

        from protein_agent.agent import ProteinAgent
        agent = ProteinAgent(email="test@example.com", _client=mock_client)
        result = agent.run("MKVL")

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(result.annotations[0].accession, "IPR001270")
        self.assertEqual(result.metadata["tool_calls"], 1)

    @patch("protein_agent.agent.InterProScanTool")
    def test_run_no_tool_call(self, MockTool):
        """LLM decides not to call the tool → empty annotations."""
        tool_instance = MagicMock()
        MockTool.return_value = tool_instance

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_stop_response()

        from protein_agent.agent import ProteinAgent
        agent = ProteinAgent(email="test@example.com", _client=mock_client)
        result = agent.run("MKVL")

        self.assertEqual(len(result.annotations), 0)
        self.assertEqual(result.metadata["tool_calls"], 0)


# ── SiteAnnotation.residue_set ────────────────────────────────────────────


class SiteAnnotationTests(unittest.TestCase):
    def test_residue_set_single(self):
        ann = SiteAnnotation("IPR001", "X", "ACTIVE_SITE", ((10, 15),))
        self.assertEqual(ann.residue_set(), frozenset(range(10, 16)))

    def test_residue_set_multi(self):
        ann = SiteAnnotation("IPR001", "X", "ACTIVE_SITE", ((10, 12), (20, 21)))
        expected = frozenset(range(10, 13)) | frozenset(range(20, 22))
        self.assertEqual(ann.residue_set(), expected)


# ---------------------------------------------------------------------------
# evaluation_agent tests
# ---------------------------------------------------------------------------


class ParsePositionsTests(unittest.TestCase):
    def test_single(self):
        result = _parse_positions("10", "20")
        self.assertEqual(result, ((10, 20),))

    def test_multi(self):
        result = _parse_positions("10|50", "20|60")
        self.assertEqual(result, ((10, 20), (50, 60)))

    def test_mismatched_raises(self):
        with self.assertRaises(ValueError):
            _parse_positions("10|20", "30")

    def test_start_gt_end_raises(self):
        with self.assertRaises(ValueError):
            _parse_positions("30", "10")


class LoadExamplesTests(unittest.TestCase):
    def _write_csv(self, rows: list[dict]) -> io.StringIO:
        buf = io.StringIO()
        fieldnames = ["uid", "seq_full", "interpro_id", "interpro_label", "start", "end"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        buf.seek(0)
        return buf

    def test_single_fragment(self):
        import tempfile, os
        rows = [{"uid": "u1", "seq_full": "MKVL", "interpro_id": "IPR000001",
                 "interpro_label": "0", "start": "1", "end": "4"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            tmp = f.name
        try:
            examples = load_examples(tmp)
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].uid, "u1")
            self.assertEqual(examples[0].fragment_parts, ((1, 4),))
        finally:
            os.unlink(tmp)

    def test_multi_fragment(self):
        import tempfile, os
        rows = [{"uid": "u2", "seq_full": "MKVLAAAA", "interpro_id": "IPR000002",
                 "interpro_label": "1", "start": "1|5", "end": "4|8"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            tmp = f.name
        try:
            examples = load_examples(tmp)
            self.assertEqual(examples[0].fragment_parts, ((1, 4), (5, 8)))
        finally:
            os.unlink(tmp)


class ResidueMetricsTests(unittest.TestCase):
    def test_perfect_match(self):
        residues = frozenset(range(1, 11))
        m = residue_metrics(residues, residues)
        self.assertEqual(m["tp"], 10)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_no_overlap(self):
        true = frozenset(range(1, 6))
        pred = frozenset(range(10, 15))
        m = residue_metrics(true, pred)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fp"], 5)
        self.assertEqual(m["fn"], 5)
        self.assertAlmostEqual(m["precision"], 0.0)  # 0/(0+5)
        self.assertAlmostEqual(m["recall"], 0.0)     # 0/(0+5)
        self.assertAlmostEqual(m["f1"], 0.0)

    def test_partial_overlap(self):
        true = frozenset(range(1, 11))   # 1..10
        pred = frozenset(range(6, 16))   # 6..15
        m = residue_metrics(true, pred)
        self.assertEqual(m["tp"], 5)
        self.assertEqual(m["fp"], 5)
        self.assertEqual(m["fn"], 5)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)


class FragmentMetricsTests(unittest.TestCase):
    def test_perfect_match(self):
        parts = ((10, 20),)
        m = fragment_metrics(parts, parts, iou_threshold=0.5)
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)

    def test_no_overlap(self):
        true = ((10, 20),)
        pred = ((50, 60),)
        m = fragment_metrics(true, pred, iou_threshold=0.5)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 1)

    def test_partial_overlap_above_threshold(self):
        true = ((1, 10),)   # residues 1..10 (10 residues)
        pred = ((6, 15),)   # residues 6..15, overlap 6..10 (5/15 ≈ 0.33) → below 0.5
        m = fragment_metrics(true, pred, iou_threshold=0.5)
        self.assertEqual(m["tp"], 0)

    def test_partial_overlap_below_threshold_passes_at_lower(self):
        true = ((1, 10),)
        pred = ((6, 15),)
        m = fragment_metrics(true, pred, iou_threshold=0.1)
        self.assertEqual(m["tp"], 1)

    def test_multiple_fragments(self):
        true = ((1, 10), (20, 30))
        pred = ((1, 10), (20, 30))
        m = fragment_metrics(true, pred, iou_threshold=0.5)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)


class AggregateResultsTests(unittest.TestCase):
    def _make_result(
        self,
        uid: str,
        label_found: bool,
        r_tp: int = 0, r_fp: int = 0, r_fn: int = 0,
        f_tp: int = 0, f_fp: int = 0, f_fn: int = 0,
        error: str | None = None,
    ) -> EvalResult:
        return EvalResult(
            uid=uid,
            interpro_id="IPR000001",
            label_found=label_found,
            residue_tp=r_tp, residue_fp=r_fp, residue_fn=r_fn,
            fragment_tp=f_tp, fragment_fp=f_fp, fragment_fn=f_fn,
            error=error,
        )

    def test_all_found(self):
        results = [
            self._make_result("a", True, r_tp=10, r_fp=0, r_fn=0, f_tp=1, f_fp=0, f_fn=0),
            self._make_result("b", True, r_tp=8,  r_fp=2, r_fn=2, f_tp=1, f_fp=0, f_fn=0),
        ]
        summary = aggregate_results(results)
        self.assertEqual(summary["n_total"], 2)
        self.assertEqual(summary["n_errors"], 0)
        self.assertAlmostEqual(summary["label_recall"], 1.0)
        self.assertAlmostEqual(summary["mean_fragment_recall"], 1.0)

    def test_with_error(self):
        results = [
            self._make_result("a", True, r_tp=10, r_fp=0, r_fn=0, f_tp=1),
            self._make_result("b", False, error="timeout"),
        ]
        summary = aggregate_results(results)
        self.assertEqual(summary["n_errors"], 1)
        self.assertAlmostEqual(summary["label_recall"], 0.5)

    def test_empty(self):
        summary = aggregate_results([])
        self.assertEqual(summary["n_total"], 0)
        self.assertAlmostEqual(summary["label_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
