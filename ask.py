import os

from dotenv import load_dotenv
from search import search
from groq import Groq
from cache import get_cache_key, save_to_cache

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def expand_query(question):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"Write a short hypothetical answer to this question in 2-3 sentences: {question}"
        }]
    )
    hypothetical_answer = response.choices[0].message.content
    print(f"\n--- HYPOTHETICAL ANSWER USED FOR SEARCH ---\n{hypothetical_answer}\n")
    return hypothetical_answer

def ask(question):
    # step 1 — check cache first
    cached_answer = get_cache_key(question)
    if cached_answer:
        print("\n--- ANSWER (from cache) ---")
        print(cached_answer)
        return

    # step 2 — expand query using HyDE
    expanded_query = expand_query(question)

    # step 3 — search using expanded query
    chunks = search(expanded_query)

    # step 4 — build context
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[{i+1}] (Page {chunk['page']}): {chunk['text']}\n\n"

    # step 5 — build prompt
    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
At the end of each sentence, cite the source number like [1] or [2].

Context:
{context}

Question: {question}

Answer:"""

    # step 6 — send to LLM with streaming
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    # step 7 — collect and print streamed answer
    print("\n--- ANSWER (streaming) ---")
    full_answer = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            print(token, end="", flush=True)
            full_answer += token

    # step 8 — save to cache
    save_to_cache(question, full_answer)

    # step 9 — print sources
    print("\n\n--- SOURCES ---")
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}] Page {chunk['page']}")

ask("What is this paper about?")