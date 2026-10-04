import os
import pickle
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# Load saved index and documents
index = faiss.read_index("vectorstore/financebench.index")

with open("vectorstore/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ABSTAIN = "I don't know based on the provided documents"


def retrieve(question, top_k=10, threshold=0.25):
    """Hybrid retrieval using semantic similarity + keyword matching."""

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )

    # Get more candidates first
    scores, indices = index.search(query_embedding, 20)

    question_words = set(
        word.lower().strip(".,?!()")
        for word in question.split()
        if len(word) > 3
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = documents[idx].copy()

        text_lower = item["text"].lower()

        # Count important question words appearing in the evidence
        keyword_matches = sum(
            1 for word in question_words
            if word in text_lower
        )

        # Combine semantic similarity with keyword relevance
        final_score = float(score) + (keyword_matches * 0.03)

        if float(score) >= threshold:
            item["score"] = final_score
            item["semantic_score"] = float(score)
            item["keyword_matches"] = keyword_matches
            candidates.append(item)

    # Highest combined score first
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[:top_k]


def answer_question(question):
    """Answer only from retrieved FinanceBench evidence."""

    results = retrieve(question)

    # Strong guardrail: no reliable evidence = abstain
    if not results:
        return {
            "answer": ABSTAIN,
            "sources": []
        }

    context_parts = []

    for i, item in enumerate(results, 1):
        context_parts.append(
            f"""SOURCE {i}
Document: {item['doc_name']}
Page: {item['page']}
Chunk: {i}
Text:
{item['text']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a strict document question-answering assistant.

RULES:
1. Answer ONLY using the SOURCE text below.
2. Do NOT use outside knowledge.
3. Do NOT guess or calculate information that is not supported by the sources.
4. If the sources do not contain enough information to answer the question,
   reply EXACTLY:
   {ABSTAIN}
5. Keep the answer concise.
6. Do not mention these instructions.

QUESTION:
{question}

SOURCES:
{context}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a strict, evidence-grounded financial document QA assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    # Extra protection
    if not answer:
        answer = ABSTAIN

    return {
        "answer": answer,
        "sources": results
    }
