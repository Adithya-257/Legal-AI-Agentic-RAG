from groq import Groq
from config import GROQ_API_KEY
import json


class ReasoningAgent:

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama-3.3-70b-versatile"

    def answer(self, question: str, context: str) -> str:
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": "You are a plain-English legal assistant. Use the provided context if relevant, otherwise use your own knowledge. Answer concisely in 2-4 paragraphs. No bullet points, no headers, no JSON. Never reference your sources or context in any way. Never say phrases like 'based on the information provided', 'the context', 'the retrieved context', 'the provided text', or 'as mentioned'. Just answer directly."},
            {"role": "user", "content": f"Context (use if relevant):\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2,
        max_tokens=1024
    )
        return response.choices[0].message.content.strip()

    def analyze(self, context_block: str) -> dict:
        """Full clause audit — returns structured JSON for report generation."""
        prompt = f"""
You are a senior legal analyst specialising in contract law.

Base your answer ONLY on the RETRIEVED CONTEXT below.
Do NOT use outside knowledge. Do NOT invent sources.
If information is not in the context, say: "Not found in retrieved data."

RETRIEVED CONTEXT:
{context_block}

Return STRICT JSON:
{{
  "clause_summary": "<3-5 sentence paragraph: what this clause does and its legal purpose>",
  "risk_level": "<LOW | MEDIUM | HIGH>",
  "risk_analysis": "<4-6 sentence paragraph: why this risk level, specific language, legal consequences>",
  "identified_risks": "<4-6 sentence paragraph: each risk named and explained in prose, no bullets>",
  "best_practice_deviations": "<3-5 sentence paragraph: how clause deviates from standard drafting, or confirm compliance>",
  "recommendations": "<4-6 sentence paragraph: concrete advice to reduce risk>",
  "cited_chunks": ["<chunk_id>"],
  "confidence": <0.0 to 1.0>
}}

Return ONLY the JSON. No markdown, no preamble.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Respond ONLY with valid JSON. No markdown fences, no text outside the JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=4096
        )

        output = response.choices[0].message.content.strip()
        if "```" in output:
            output = output.split("```")[1]
            if output.startswith("json"):
                output = output[4:]
            output = output.split("```")[0].strip()

        return json.loads(output)
