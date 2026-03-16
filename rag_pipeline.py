import numpy as np
from google import genai
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

docs = [
    "India is formerly known as Golden Bird",
    "Population wise, India is the lasrgest country followed by China",
    "India is 4th largest economy in the world. USA at first, China at second, Germany at third.",
    "India's GDP is around 4.3 trillion dollars",
    "Pakistan is a very poor country, with strong foundations of terrorists",
    "New Zealand is one of the country which is much far away from the world's chit-chat, having population of around 55-60 lakhs only"
]

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = model.encode(docs)

def retrieve(query, top_k=20):

    query_embedding = model.encode(query)

    scores = cosine_similarity([query_embedding], embeddings)[0]

    indices = np.argsort(scores)[::-1][:top_k]

    retrieved_docs = [docs[i] for i in indices]
    retrieved_scores = [scores[i] for i in indices]

    return retrieved_docs, retrieved_scores


def rerank(query, docs):

    pairs = [[query, doc] for doc in docs]

    scores = reranker.predict(pairs)

    indices = np.argsort(scores)[::-1]

    reranked_docs = [docs[i] for i in indices]

    return reranked_docs

def build_prompt(query, context_docs):

    context = "\n".join(context_docs)

    prompt = f"""
Answer the user's question ONLY using the context below.

Context:
{context}

Question:
{query}

If the answer is not contained in the context, say:
"I don't have enough information to answer that."
"""

    return prompt

def generate_answer(prompt):

    client = genai.Client(api_key="AIzaSyAY49bOXeW_FDV7tEX2MajZHwqnZFcqb3M")

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,          # For simple prompts, just pass the string
        config={
            "system_instruction": "Answer only using provided context."
        }
    )

    return response.text

def rag_pipeline(query):

    print("\nUser Query:", query)

    retrieved_docs, retrieved_scores = retrieve(query)

    best_score = retrieved_scores[0]

    print("\nBest similarity score:", best_score)

    # Threshold check
    if best_score < 0.7:
        return "I don't have enough information to answer that."

    # Re-rank
    reranked_docs = rerank(query, retrieved_docs)

    # Take top 5
    top_docs = reranked_docs[:5]

    prompt = build_prompt(query, top_docs)

    answer = generate_answer(prompt)

    return answer

if __name__ == "__main__":

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = rag_pipeline(query)

        print("\nAnswer:\n", answer)














