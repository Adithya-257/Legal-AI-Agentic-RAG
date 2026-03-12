"""
report_generator.py

Formats structured legal analysis into a human-readable report.
"""

import json
import textwrap


class ReportGenerator:
    """Render legal analysis results as a readable report"""

    def generate(self, analysis: dict, clause: str = "") -> str:

        def paragraph(text):
            if not text:
                return "Not available."
            return textwrap.fill(text.strip(), width=80)

        risk = analysis.get("risk_level", "Unknown")
        confidence = analysis.get("confidence", "N/A")

        clause_summary = analysis.get("clause_summary", "")
        risk_analysis = analysis.get("risk_analysis", "")
        identified_risks = analysis.get("identified_risks", "")
        deviations = analysis.get("best_practice_deviations", "")
        recommendations = analysis.get("recommendations", "")
        citations = analysis.get("cited_chunks", [])

        lines = []

        lines.append("\nLEGAL RISK ANALYSIS REPORT")
        lines.append("=" * 60)

        if clause:
            lines.append("\nCLAUSE UNDER REVIEW")
            lines.append("-" * 60)
            lines.append(paragraph(clause))

        lines.append("\nCLAUSE SUMMARY")
        lines.append("-" * 60)
        lines.append(paragraph(clause_summary))

        lines.append("\nRISK LEVEL")
        lines.append("-" * 60)
        lines.append(risk.upper())

        lines.append("\nRISK ANALYSIS")
        lines.append("-" * 60)
        lines.append(paragraph(risk_analysis))

        lines.append("\nIDENTIFIED RISKS")
        lines.append("-" * 60)
        lines.append(paragraph(identified_risks))

        lines.append("\nBEST PRACTICE DEVIATIONS")
        lines.append("-" * 60)
        lines.append(paragraph(deviations))

        lines.append("\nRECOMMENDATIONS")
        lines.append("-" * 60)
        lines.append(paragraph(recommendations))

        lines.append("\nMETADATA")
        lines.append("-" * 60)

        metadata = {
            "confidence_score": confidence,
            "cited_chunks": citations
        }

        lines.append(json.dumps(metadata, indent=4))

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
