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

##  Architecture

<img width="1102" height="287" alt="Legal ai visual copy" src="https://github.com/user-attachments/assets/36e69fb4-07a3-4bee-8758-479e37738833" />

##  Agent Roles

###  Knowledge Base Structure
The system uses four specialized vector databases, each storing a different category of legal knowledge.
Instead of querying a single large database, the LLM Router selects the most relevant databases, improving retrieval quality and reducing noise.

1. DB-A — Legal Definitions
  Contains explanations of legal terminology and contract language.

  Example content:
  
  * clause definitions
  * legal terminology explanations
  * glossary-style references
    
Purpose: Helps the system interpret legal terms appearing in clauses.

2. DB-B — Risk Patterns
   Contains examples of common contractual risks and legal red flags.
   
   Example content:
   
   * indemnification risks
   * liability clauses
   * confidentiality breaches
   * dispute resolution issues
  
Purpose: Allows the system to detect risky clause structures and potential legal exposure.

3. DB-C — Legal Standards & Best Practices

  Contains industry-standard drafting practices and regulatory guidance.

  Example content:

  * compliance requirements
  * recommended clause structures
  * legal drafting best practices

Purpose: Allows the system to compare clauses against accepted legal standards.

4. DB-D — Summary Guidelines
   Contains knowledge used for final analysis synthesis.

   Example Content:

    * risk summarization patterns
    * legal analysis templates
    * audit reasoning guidance
   
  Purpose: Helps structure the final legal analysis and recommendations.



Using multiple specialized databases enables targeted retrieval. It helps the LLM Router to select relevant knowledge bases, this helps the retriever to query only those relevant databases. Resulting in a Higher relevance context and better legal reasoning.  This approach improves retrieval precision, reasoning accuracy and scalability of the knowledge base.

###   LLM Router
Uses an LLM to determine which legal databases are relevant for a given clause.



  
