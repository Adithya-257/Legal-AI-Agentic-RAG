"""
app.py

Main CLI entrypoint for the Legal Audit Agent.
Handles user input, intent detection, and execution of the agent pipeline.
"""

import sys
import textwrap

from graph.agent_graph import graph
from agents.retriever import DatabaseRetriever
from agents.augmentor import ContextAugmentor
from agents.reasoning_agent import ReasoningAgent
from agents.report_generator import ReportGenerator


QUESTION_STARTERS = (
    "what", "how", "who", "when", "why", "where",
    "explain", "define", "describe", "tell me",
    "is ", "are ", "can ", "does ", "do "
)


def is_question(text: str) -> bool:
    """Detect if input is a question."""
    t = text.strip().lower()
    return t.endswith("?") or t.startswith(QUESTION_STARTERS)


def print_banner():
    print("\n" + "=" * 60)
    print("LEGAL AUDIT AGENT — Agentic RAG System")
    print("=" * 60 + "\n")


def get_input(prompt: str) -> str:
    """Read multi-line user input."""
    print(prompt)
    print("(Press Enter on an empty line to submit)\n")

    lines = []

    while True:
        try:
            line = input("> ")
        except EOFError:
            break

        if line.strip() == "" and lines:
            break

        lines.append(line)

    return " ".join(lines).strip()


# ─────────────────────────────────────────
# Question Answering Flow
# ─────────────────────────────────────────

def run_qa(question: str):

    print("\nDetected: Question")
    print("-" * 60)

    retriever = DatabaseRetriever(vector_dbs_path="Vector_DBs")
    augmentor = ContextAugmentor()
    agent = ReasoningAgent()

    results = retriever.retrieve(
        query=question,
        databases=["db_a_defs", "db_b_risks", "db_c_standards"],
        top_k=3
    )

    context = augmentor.build_simple_context(results)
    answer = agent.answer(question, context)

    print("\nANSWER")
    print("-" * 60)

    wrapped = textwrap.fill(
        answer,
        width=80,
        initial_indent="  ",
        subsequent_indent="  "
    )

    print(wrapped)
    print()


# ─────────────────────────────────────────
# Clause Audit Flow
# ─────────────────────────────────────────

def run_audit(clause: str):

    print("\nRunning clause audit...")
    print("-" * 60)

    initial_state = {
        "clause": clause,
        "step_times": {},
        "errors": []
    }

    result = graph.invoke(initial_state)

    reporter = ReportGenerator()
    analysis = result.get("analysis", {})

    report = reporter.generate(analysis, clause=clause)
    print(report)

    # Performance summary
    times = result.get("step_times", {})

    if times:
        print("\nPerformance")
        print("-" * 60)

        total = sum(times.values())

        for step, duration in times.items():
            print(f"{step:<25} {duration:.3f}s")

        print(f"{'TOTAL':<25} {total:.3f}s\n")

    # Error reporting
    if result.get("errors"):
        print("Errors:")
        for err in result["errors"]:
            print(f" - {err}")
        print()


# ─────────────────────────────────────────
# Minimal CLI Menu
# ─────────────────────────────────────────

def menu():

    print("Select an option:\n")
    print("1. Ask a legal question")
    print("2. Audit a contract clause")
    print("q. Quit\n")

    while True:
        choice = input("> ").strip().lower()

        if choice in ("1", "2", "q"):
            return choice

        print("Please enter 1, 2, or q.")


# ─────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────

if __name__ == "__main__":

    print_banner()

    while True:

        choice = menu()

        if choice == "q":
            print("\nGoodbye.\n")
            sys.exit(0)

        elif choice == "1":
            question = get_input("Enter your legal question:")
            if question:
                run_qa(question)

        elif choice == "2":
            clause = get_input("Enter the contract clause to audit:")
            if clause:
                run_audit(clause)

        again = input("Run again? (y/n): ").strip().lower()

        if again != "y":
            print("\nGoodbye.\n")
            break
