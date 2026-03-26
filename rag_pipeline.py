import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()

docs = [                                                    # This is a short n basic Knowledge Base
    "India is formerly known as Golden Bird",
    "Population wise, India is the lasrgest country followed by China",
    "India is 4th largest economy in the world. USA at first, China at second, Germany at third.",
    "India's GDP is around 4.3 trillion dollars",
    "Pakistan is a very poor country, with strong foundations of terrorists",
    "New Zealand is one of the country which is much far away from the world's chit-chat, having population of around 55-60 lakhs only"
]

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = model.encode(docs)                     # Each document in the KB will be converted into vector

def retrieve(query, top_k=20):

    query_embedding = model.encode(query)           # Jo query aayi h, use embedding me convert krdiya

    scores = cosine_similarity([query_embedding], embeddings)[0]            # Measuring cosine similarity of the query embedding, with the rest of embeedings in KB

    indices = np.argsort(scores)[::-1][:top_k]                              # Sorting the docs, jyada score comes first

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

    client = genai.Client(os.getenv("API_KEY"))

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,          
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

    if best_score < 0.7:                                            # Threshold value is 0.7
        return "I don't have enough information to answer that."

    reranked_docs = rerank(query, retrieved_docs)                   # Re ranking ho rhi h

    top_docs = reranked_docs[:5]                                    # Top 5 docs are returned for the context

    prompt = build_prompt(query, top_docs)                          # Prompt is generated to sent it to LLM

    answer = generate_answer(prompt)

    return answer

if __name__ == "__main__":

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = rag_pipeline(query)

        print("\nAnswer is:\n", answer)
