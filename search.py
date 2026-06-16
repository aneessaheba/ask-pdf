import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# load the embedding model and reranker
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# load the database
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection(name="pdf_chunks")

# load all chunks from ChromaDB for BM25
all_data = collection.get()
all_texts = all_data["documents"]
all_metadatas = all_data["metadatas"]

# build BM25 index
tokenized_chunks = [text.split() for text in all_texts]
bm25 = BM25Okapi(tokenized_chunks)

def rrf_fusion(vector_ids, bm25_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def search(query):
    # --- vector search ---
    query_vector = model.encode(query).tolist()
    vector_results = collection.query(
        query_embeddings=[query_vector],
        n_results=10
    )
    vector_ids = vector_results["ids"][0]

    # --- BM25 keyword search ---
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    bm25_ids = [f"chunk_{i}" for i in top_bm25_indices]

    # --- merge using RRF ---
    merged_ids = rrf_fusion(vector_ids, bm25_ids)[:10]

    # --- re-rank using cross encoder ---
    pairs = [[query, all_texts[int(chunk_id.split("_")[1])]] for chunk_id in merged_ids]
    rerank_scores = reranker.predict(pairs)

    # sort by rerank score
    ranked = sorted(zip(merged_ids, rerank_scores), key=lambda x: x[1], reverse=True)

    # keep top 7 after reranking
    chunks = []
    for chunk_id, score in ranked[:7]:
        index = int(chunk_id.split("_")[1])
        chunks.append({
            "text": all_texts[index],
            "page": all_metadatas[index]["page"]
        })

    return chunks