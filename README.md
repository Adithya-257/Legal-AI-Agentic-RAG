# Legal AI-Agentic RAG

Multi-Agent Legal Contract Analyzer using LLM Routing, Vector Databases, and LangGraph

![Python](https://img.shields.io/badge/Python-3.14-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20AI-purple)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-green)


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

Example decision:

Clause type: indemnification
Databases selected:
- Risk Patterns
- Legal Standards

###   Vector Retriever

Retrieves relevant legal text chunks from ChromaDB vector databases using semantic search.

Each chunk contains:

* legal explanation
* risk pattern
* best practice reference

###   Context Augmentor

Builds a structured context block combining:

* the clause under review
* retrieved legal knowledge
* metadata (source + similarity)

###   Reasoning Agent

Uses Groq LLMs to perform a grounded legal analysis.

Outputs structured JSON including:

* clause summary
* risk level
* identified risks
* compliance deviations
* recommendations
* cited evidence

### Report Generator

Converts the structured output into a human-readable legal audit report.

## Tech Stack

- Python
- LangGraph — agent orchestration
- Groq API — LLM inference
- ChromaDB — vector database
- Sentence Transformers — embeddings
- HuggingFace Models
- Retrieval-Augmented Generation (RAG)

## Features

- Multi-agent LLM architecture
- Intelligent LLM-based routing
- Multi-database vector retrieval
- Context-grounded legal reasoning
- Evidence-based analysis with citations
- Structured legal audit reports
- CLI interface for interactive analysis


##  WORKING EXAMPLE
###   Input

<img width="289" height="95" alt="image" src="https://github.com/user-attachments/assets/ad7e357e-e30c-43e2-97bf-e6e1ae4b5346" />

<img width="229" height="58" alt="image" src="https://github.com/user-attachments/assets/571c99be-cd46-4a03-b781-75ce3a707d5c" />

###   Output

<img width="336" height="104" alt="image" src="https://github.com/user-attachments/assets/d4aa0ebf-7fe9-4bfe-9f9f-c1b0fa5c0bcb" />

 Force majeure is a legal concept that refers to an  unforeseen event or circumstance that prevents a partyfrom fulfilling  
 their contractual obligations. It is often defined as an extraordinary and unforeseeable event that is beyond the control 
 of the party invoking it, such as a natural disaster, war, or other major disruption. When a force majeure event occurs,  
 it can excuse a party from performing their contractual duties, at least temporarily, without being held liable for breach 
 of contract.  The concept of force majeure is typically included in  contracts as a clause that outlines the specific  
 circumstances under which a party may be excused from performing their obligations. This clause is intended to provide a 
 way for parties to allocate risk and responsibility in the event of unforeseen circumstances.  By including a force 
 majeure clause, parties can ensure that they are not held responsible for events that are beyond their control.  In 
 general, force majeure clauses are designed to provide a fair and reasonable way to handle unexpected events that may 
 affect the performance of a contract. They can help to prevent disputes and litigation by providing a clear framework for 
 dealing with unforeseen circumstances. However, the specific terms and conditions of a force majeure clause can vary 
 widely depending on the contract and the parties involved.  It's worth noting that force majeure is often confused with 
 the term "act of God," which refers to a natural disaster or other event that is beyond human control. While the two terms 
 are related, they are not exactly the same thing. Force majeure can include a broader range of events, including human 
 actions such as war or terrorism, in addition to natural disasters.


###   CLAUSE EXAMPLE

INPUT CLAUSE:-
The Contractor shall not be liable for any damages, losses,
or claims of any kind arising from the performance of this Agreement.


<img width="335" height="84" alt="image" src="https://github.com/user-attachments/assets/97ee13a2-8126-4403-8f85-b23caa76e18f" />


##  Legal Risk Analysis Report

* Clause Summary

The clause under review is a liability exemption clause which states that the Contractor shall not be liable for any damages, losses, or claims arising from the performance of the agreement. The purpose of this clause is to protect the contractor from financial exposure and allocate risk between the contracting parties. However, the language is broadly drafted and may create uncertainty in interpretation. In practice, such broad exclusions of liability can raise concerns regarding fairness and enforceability. Courts may scrutinize these clauses if they appear to exempt a party from responsibility for negligence or misconduct.

* Risk Level
  
  [HIGH]

* Risk Analysis
  
The clause presents a high level of risk due to its sweeping liability exclusion. By stating that the contractor shall not  be liable for any damages, losses, or claims of any kind, the clause removes virtually all responsibility from the contractor without defining clear limits or exceptions. Such language may unintentionally exempt the contractor even in cases of negligence or misconduct. This imbalance may be considered unreasonable or unenforceable depending on jurisdiction. Additionally, courts may interpret such clauses narrowly if they conflict with statutory obligations or public policy protections.

* Identified Risks
  
Several risks arise from this clause. First, the extremely broad language introduces ambiguity and increases the likelihood of disputes over interpretation. Second, the clause may unintentionally shield the contractor from liability for negligent or improper performance. Third, the clause may be challenged legally if it appears to unfairly shift all risk to the other contracting party. Similar liability waivers have historically been invalidated when they were considered overly broad or inconsistent with statutory protections. Finally, the clause may create enforcement risks if it conflicts with mandatory legal obligations.

* Best Practice Deviations
  
The clause deviates from standard legal drafting practices by failing to include reasonable limitations or carve-outs. Best practice clauses typically exclude liability only for specific categories of damages or include exceptions for gross negligence, fraud, or wilful misconduct. Additionally, well-drafted contracts aim to maintain balanced risk allocation between parties. The current wording may therefore be viewed as disproportionate and inconsistent with widely accepted contractual standards.

* Recommendations
  
To reduce legal risk, the clause should be narrowed to include clearly defined limitations. Exceptions should be added for gross negligence, fraud, or wilful misconduct to ensure the clause remains enforceable. Liability caps may also be introduced to balance protection while maintaining fairness between parties. It is also advisable to align the clause with standard contractual language commonly used in commercial agreements. Finally, the clause should be reviewed in the context of the entire contract to ensure consistency with the agreement’s overall risk allocation framework.

* Metadata
  
{
  "confidence_score": 0.8,
  "cited_chunks": [
    "DB_B_RISKS-1",
    "DB_B_RISKS-2",
    "DB_B_RISKS-3",
    "DB_C_STANDARDS-1",
    "DB_C_STANDARDS-2",
    "DB_C_STANDARDS-3"
  ]
}


## Future Improvements

- Web UI using Streamlit or FastAPI
- Support for full contract PDF analysis
- Automatic clause extraction
- Multi-jurisdiction legal knowledge bases
- Improved risk scoring models


