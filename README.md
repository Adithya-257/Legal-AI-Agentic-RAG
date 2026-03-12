# Legal AI-Agentic RAG

Multi-Agent Legal Contract Analyzer using LLM Routing, Vector Databases, and LangGraph

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20AI-purple)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project implements an **Agentic Retrieval-Augmented Generation (RAG) system** for analyzing legal contract clauses.

Instead of relying on a single prompt, the system uses a multi-agent pipeline where specialized agents handle different stages of the analysis:

* LLM Router decides which legal knowledge bases are relevant.
  
* Vector Retriever fetches supporting legal context.

* Context Augmentor builds structured context.
  
* Reasoning Agent performs legal analysis using an LLM.
  
* Report Generator produces a structured risk report.

The system uses **LangGraph for agent orchestration, Groq LLMs for reasoning, and ChromaDB** for vector retrieval.

The result is a **grounded legal analysis** system that cites evidence from retrieved documents instead of hallucinating answers.

#  Architecture



  
