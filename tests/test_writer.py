"""Unit tests for the Writer's Markdown rendering (``agents.writer.to_markdown``).

Verifies the rendered structure: the H1 title, the executive summary, one H2
per section (in order), and the trailing References section with 1-based
numbered source URLs — plus the stub report and the no-citations case.
"""

from __future__ import annotations

import os

# config.py reads the environment at import time — these must be set before any
# project module is imported (load_dotenv() never overrides existing vars).
os.environ["MOCK_MODE"] = "true"
os.environ["QDRANT_PATH"] = ":memory:"

import unittest

from agents.writer import _empty_report, to_markdown
from schemas.models import FinalReport, ReportSection


def _report() -> FinalReport:
    """A small two-section report with two citations."""
    return FinalReport(
        title="Quantum Computing in 2026",
        summary="A short executive summary of the field [1].",
        sections=[
            ReportSection(
                heading="Background",
                content="Qubits and decoherence basics [1].",
                citations=["https://example.org/qubits"],
            ),
            ReportSection(
                heading="Recent Advances",
                content="Error-correction milestones [2].",
                citations=["https://example.org/error-correction"],
            ),
        ],
        all_citations=[
            "https://example.org/qubits",
            "https://example.org/error-correction",
        ],
    )


class ToMarkdownTest(unittest.TestCase):
    """Structure of the rendered Markdown report."""

    def test_title_is_h1_on_first_line(self) -> None:
        markdown = to_markdown(_report())
        self.assertEqual(markdown.splitlines()[0], "# Quantum Computing in 2026")

    def test_summary_appears_before_first_section(self) -> None:
        markdown = to_markdown(_report())
        self.assertIn("A short executive summary of the field [1].", markdown)
        self.assertLess(
            markdown.index("A short executive summary"), markdown.index("## Background")
        )

    def test_sections_render_as_h2_in_order_with_content(self) -> None:
        markdown = to_markdown(_report())
        background = markdown.index("## Background")
        advances = markdown.index("## Recent Advances")
        self.assertLess(background, advances)
        self.assertIn("Qubits and decoherence basics [1].", markdown)
        self.assertIn("Error-correction milestones [2].", markdown)

    def test_references_section_lists_numbered_citations_last(self) -> None:
        markdown = to_markdown(_report())
        references = markdown.index("## References")
        self.assertLess(markdown.index("## Recent Advances"), references)
        lines = markdown.splitlines()
        self.assertIn("1. https://example.org/qubits", lines)
        self.assertIn("2. https://example.org/error-correction", lines)
        # Citation numbering follows the all_citations order.
        self.assertLess(
            lines.index("1. https://example.org/qubits"),
            lines.index("2. https://example.org/error-correction"),
        )

    def test_output_ends_with_single_trailing_newline(self) -> None:
        markdown = to_markdown(_report())
        self.assertTrue(markdown.endswith("\n"))
        self.assertFalse(markdown.endswith("\n\n"))

    def test_no_references_section_without_citations(self) -> None:
        report = _report()
        report.all_citations = []
        markdown = to_markdown(report)
        self.assertNotIn("## References", markdown)

    def test_empty_report_stub_renders_status_section(self) -> None:
        markdown = to_markdown(_empty_report("an unanswerable query"))
        self.assertTrue(markdown.startswith("# Research Report: an unanswerable query"))
        self.assertIn("## Status", markdown)
        self.assertNotIn("## References", markdown)


if __name__ == "__main__":
    unittest.main()
