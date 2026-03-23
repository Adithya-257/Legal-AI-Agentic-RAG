import streamlit as st
from graph.agent_graph import graph
from agents.report_generator import ReportGenerator

st.set_page_config(page_title="Legal Audit Agent", page_icon="⚖️")
st.title("⚖️ Legal AI Audit Agent")
st.caption("Powered by Agentic RAG · LangGraph · LLaMA-3.3-70B")

clause = st.text_area("Enter a legal clause to audit:", height=150)

if st.button("Run Audit"):
    if not clause.strip():
        st.warning("Please enter a clause.")
    else:
        with st.spinner("Running agentic analysis..."):
            result = graph.invoke({
                "clause": clause,
                "step_times": {},
                "errors": []
            })
            analysis = result.get("analysis", {})
            reporter = ReportGenerator()
            report = reporter.generate(analysis, clause=clause)
        
        st.success("Analysis complete.")
        st.markdown("### 📋 Audit Report")
        st.markdown(report)

        if result.get("errors"):
            st.error("Errors: " + ", ".join(result["errors"]))
