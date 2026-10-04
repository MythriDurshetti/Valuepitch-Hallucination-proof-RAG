# ValuePitch – Hallucination-Proof Document Q&A Bot

## Overview
A Retrieval-Augmented Generation (RAG) system built for the ValuePitch AI Hackathon using the FinanceBench dataset.

The system answers financial questions only from the provided document evidence and provides source citations.

## Key Features

- FinanceBench document evidence retrieval
- FAISS vector search
- Sentence Transformer embeddings
- Hybrid semantic + keyword retrieval
- Groq LLM for answer generation
- Source citations with document name and page
- Hallucination prevention
- Exact abstention when evidence is insufficient

## Hallucination Guardrail

If the required information cannot be supported by the retrieved documents, the system returns:

> I don't know based on the provided documents

This prevents the model from answering using outside knowledge.

## Architecture

User Question  
↓  
Sentence Transformer Embedding  
↓  
FAISS + Keyword Retrieval  
↓  
Relevant Evidence  
↓  
Groq LLM  
↓  
Answer + Source Citation / Abstention

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Hugging Face Datasets
- Groq
- PyPDF
- python-dotenv

## Example

Question:
What is the FY2018 capital expenditure amount for 3M?

Answer:
$1,577 million

Source:
3M_2018_10K — Page 59

## Run Locally

```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
