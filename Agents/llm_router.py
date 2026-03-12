"""
llm_router.py

LLM-based routing agent that decides which vector databases to query.
"""

from typing import Dict
from dotenv import load_dotenv
import os
import json
from groq import Groq


class MixtralRouter:
    """Routes legal clauses to relevant knowledge bases using an LLM"""

    def __init__(self):

        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key.strip())
        self.model = "llama-3.3-70b-versatile"

        self.db_descriptions = {
            "db_a_defs": "Legal term definitions and glossaries",
            "db_b_risks": "Risk patterns, red flags, and liability issues",
            "db_c_standards": "Best practices, compliance standards, and regulations",
            "db_d_summary": "Summary synthesis rules and aggregation guidelines"
        }

    def route(self, clause_text: str) -> Dict:
        """Determine which databases should be queried for a clause."""

        prompt = self._build_routing_prompt(clause_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal routing agent. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=400
            )

            response_text = response.choices[0].message.content.strip()

            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text

            decision = json.loads(json_str)

            return {
                "databases": decision.get("databases", []),
                "reasoning": decision.get("reasoning", ""),
                "clause_type": decision.get("clause_type", "unknown"),
                "confidence": decision.get("confidence", "medium"),
                "clause_preview": clause_text[:100]
            }

        except Exception:

            return {
                "databases": ["db_a_defs", "db_b_risks", "db_c_standards"],
                "reasoning": "Fallback routing due to LLM error",
                "clause_type": "unknown",
                "confidence": "low",
                "clause_preview": clause_text[:100]
            }

    def _build_routing_prompt(self, clause_text: str) -> str:
        """Construct routing prompt."""

        return f"""
Analyze this legal clause and decide which knowledge bases to query.

CLAUSE:
{clause_text}

AVAILABLE DATABASES:
- db_a_defs: {self.db_descriptions["db_a_defs"]}
- db_b_risks: {self.db_descriptions["db_b_risks"]}
- db_c_standards: {self.db_descriptions["db_c_standards"]}
- db_d_summary: {self.db_descriptions["db_d_summary"]}

Return JSON only in the following format:

{{
  "clause_type": "definition | indemnification | compliance | summary | other",
  "databases": ["db_b_risks", "db_c_standards"],
  "reasoning": "Brief explanation",
  "confidence": "high | medium | low"
}}
"""
