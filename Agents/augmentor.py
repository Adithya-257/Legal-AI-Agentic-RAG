"""
augmentor.py

Builds structured context from retrieved vector database chunks
for LLM analysis.
"""

from typing import Dict, List


class ContextAugmentor:
    """Formats retrieved knowledge into LLM-ready context"""

    def __init__(self):

        self.db_names = {
            "db_a_defs": "Legal Definitions",
            "db_b_risks": "Risk Patterns",
            "db_c_standards": "Best Practices & Standards",
            "db_d_summary": "Summary Guidelines"
        }

    def build_context(
        self,
        clause: str,
        retrieval_results: Dict[str, List[Dict]],
        include_metadata: bool = True
    ) -> str:
        """Create structured context for the LLM."""

        context = self._build_header(clause)
        context += self._build_retrieved_sections(retrieval_results, include_metadata)
        context += self._build_footer()

        return context

    def _build_header(self, clause: str) -> str:
        """Header containing the clause being analyzed."""

        return (
            "LEGAL CLAUSE ANALYSIS CONTEXT\n"
            "========================================\n\n"
            "CLAUSE UNDER REVIEW:\n"
            "----------------------------------------\n"
            f"{clause.strip()}\n"
            "----------------------------------------\n\n"
        )

    def _build_retrieved_sections(
        self,
        retrieval_results: Dict[str, List[Dict]],
        include_metadata: bool
    ) -> str:
        """Insert retrieved knowledge chunks."""

        context = "RETRIEVED KNOWLEDGE BASE\n"
        context += "========================================\n\n"

        for db_id, chunks in retrieval_results.items():

            if not chunks:
                continue

            db_name = self.db_names.get(db_id, db_id)

            context += f"{db_name.upper()}\n"
            context += "----------------------------------------\n"

            for i, chunk in enumerate(chunks, 1):

                context += f"[{db_id.upper()}-{i}]\n"
                context += chunk["text"].strip() + "\n"

                if include_metadata:
                    source = chunk["metadata"].get("source", "Unknown")
                    similarity = chunk["similarity"]

                    context += f"Source: {source}\n"
                    context += f"Relevance: {similarity:.2%}\n"

                context += "\n"

        return context

    def _build_footer(self) -> str:
        """Instructions for the reasoning agent."""

        return (
            "========================================\n"
            "ANALYSIS INSTRUCTIONS\n"
            "========================================\n\n"
            "1. Identify legal risks or red flags\n"
            "2. Check compliance with best practices\n"
            "3. Explain relevant definitions\n"
            "4. Provide recommendations if needed\n"
            "5. Cite sources using [DB_ID-N]\n"
        )
