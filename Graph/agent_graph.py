"""
agent_graph.py
LangGraph orchestration for the Legal Audit Agent pipeline.

Pipeline:
Routing → Retrieval → Context Augmentation → Reasoning
"""

from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
import time

from agents.llm_router import MixtralRouter
from agents.retriever import DatabaseRetriever
from agents.augmentor import ContextAugmentor
from agents.reasoning_agent import ReasoningAgent


# -------------------------------------------------------------------
# STATE DEFINITION
# -------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Shared state across the agent workflow.
    """

    clause: str

    routing: dict
    routing_confidence: str

    retrieved: dict
    retrieval_stats: dict

    context: str
    context_tokens: int

    analysis: dict

    step_times: dict
    errors: List[str]


# -------------------------------------------------------------------
# AGENT INITIALIZATION
# -------------------------------------------------------------------

router = MixtralRouter()
retriever = DatabaseRetriever(vector_dbs_path="Vector_DBs")
augmentor = ContextAugmentor()
reasoner = ReasoningAgent()


# -------------------------------------------------------------------
# NODE: ROUTING
# -------------------------------------------------------------------

def routing_node(state: AgentState) -> AgentState:
    """LLM routing: decides which vector DBs to query."""
    start = time.time()

    try:
        decision = router.route(state["clause"])
        elapsed = time.time() - start

        return {
            "routing": decision,
            "routing_confidence": decision.get("confidence", "medium"),
            "step_times": {**state.get("step_times", {}), "routing": elapsed}
        }

    except Exception as e:
        return {
            "routing": {"databases": ["db_a_defs", "db_b_risks", "db_c_standards"]},
            "routing_confidence": "low",
            "errors": state.get("errors", []) + [f"Routing error: {str(e)}"]
        }


# -------------------------------------------------------------------
# NODE: RETRIEVAL
# -------------------------------------------------------------------

def retrieval_node(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from selected vector DBs."""
    start = time.time()

    try:
        results = retriever.retrieve(
            query=state["clause"],
            databases=state["routing"]["databases"],
            top_k=3
        )

        elapsed = time.time() - start
        total_chunks = sum(len(chunks) for chunks in results.values())

        return {
            "retrieved": results,
            "retrieval_stats": {
                "total_chunks": total_chunks,
                "databases_queried": len(results),
                "retrieval_time": elapsed
            },
            "step_times": {**state.get("step_times", {}), "retrieval": elapsed}
        }

    except Exception as e:
        return {
            "retrieved": {},
            "retrieval_stats": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Retrieval error: {str(e)}"]
        }


# -------------------------------------------------------------------
# NODE: CONTEXT AUGMENTATION
# -------------------------------------------------------------------

def augmentation_node(state: AgentState) -> AgentState:
    """Build LLM context using retrieved chunks."""
    start = time.time()

    try:
        context = augmentor.build_context(
            clause=state["clause"],
           
