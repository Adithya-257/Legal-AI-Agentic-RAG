"""
agent_graph.py
LangGraph orchestration for Agentic Legal RAG pipeline
"""

from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
import time

from Agents.llm_router import MixtralRouter
from Agents.retriever import DatabaseRetriever
from Agents.augmentor import ContextAugmentor
from Agents.reasoning_agent import ReasoningAgent


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
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


# ─────────────────────────────────────────────────────────────────────────────
# INIT AGENTS
# ─────────────────────────────────────────────────────────────────────────────

router = MixtralRouter()
retriever = DatabaseRetriever(vector_dbs_path="Vector_DBs")
augmentor = ContextAugmentor()
reasoner = ReasoningAgent()


# ─────────────────────────────────────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────────────────────────────────────

def routing_node(state: AgentState) -> AgentState:
    start = time.time()
    try:
        decision = router.route(state["clause"])
        return {
            "routing": decision,
            "routing_confidence": decision.get("confidence", "medium"),
            "step_times": {**state.get("step_times", {}), "routing": time.time() - start}
        }
    except Exception as e:
        return {
            "routing": {"databases": ["db_a_defs", "db_b_risks", "db_c_standards"]},
            "routing_confidence": "low",
            "errors": state.get("errors", []) + [f"Routing error: {str(e)}"]
        }


def retrieval_node(state: AgentState) -> AgentState:
    start = time.time()
    try:
        results = retriever.retrieve(
            query=state["clause"],
            databases=state["routing"]["databases"],
            top_k=3
        )

        total_chunks = sum(len(chunks) for chunks in results.values())

        return {
            "retrieved": results,
            "retrieval_stats": {
                "total_chunks": total_chunks,
                "databases_queried": len(results),
                "retrieval_time": time.time() - start
            },
            "step_times": {**state.get("step_times", {}), "retrieval": time.time() - start}
        }
    except Exception as e:
        return {
            "retrieved": {},
            "retrieval_stats": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Retrieval error: {str(e)}"]
        }


def augmentation_node(state: AgentState) -> AgentState:
    start = time.time()
    try:
        context = augmentor.build_context(
            clause=state["clause"],
            retrieval_results=state["retrieved"]
        )

        return {
            "context": context,
            "context_tokens": len(context) // 4,
            "step_times": {**state.get("step_times", {}), "augmentation": time.time() - start}
        }
    except Exception as e:
        return {
            "context": f"Error building context: {str(e)}",
            "context_tokens": 0,
            "errors": state.get("errors", []) + [f"Augmentation error: {str(e)}"]
        }


def reasoning_node(state: AgentState) -> AgentState:
    start = time.time()
    try:
        result = reasoner.analyze(state["context"])
        return {
            "analysis": result,
            "step_times": {**state.get("step_times", {}), "reasoning": time.time() - start}
        }
    except Exception as e:
        return {
            "analysis": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Reasoning error: {str(e)}"]
        }


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGES
# ─────────────────────────────────────────────────────────────────────────────

def should_retrieve(state: AgentState) -> Literal["retrieve", "skip"]:
    if not state.get("routing", {}).get("databases") or state.get("routing_confidence") == "very_low":
        return "skip"
    return "retrieve"


def needs_fallback(state: AgentState) -> Literal["augment", "fallback"]:
    total = sum(len(chunks) for chunks in state.get("retrieved", {}).values())
    return "fallback" if total == 0 else "augment"


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def fallback_node(state: AgentState) -> AgentState:
    context = f"""
CLAUSE:
{state['clause']}

NOTE: No relevant knowledge retrieved. Analyze using general legal reasoning.
"""
    return {
        "context": context,
        "context_tokens": len(context) // 4,
        "errors": state.get("errors", []) + ["No retrieval results"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("route", routing_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("augment", augmentation_node)
    g.add_node("fallback", fallback_node)
    g.add_node("reason", reasoning_node)

    g.set_entry_point("route")

    g.add_conditional_edges("route", should_retrieve, {
        "retrieve": "retrieve",
        "skip": "fallback"
    })

    g.add_conditional_edges("retrieve", needs_fallback, {
        "augment": "augment",
        "fallback": "fallback"
    })

    g.add_edge("augment", "reason")
    g.add_edge("fallback", "reason")
    g.add_edge("reason", END)

    return g.compile()


graph = build_graph()
