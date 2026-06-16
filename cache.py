import redis
import json
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np

# connect to Redis running on your laptop
r = redis.Redis(host="localhost", port=6379, db=0)

# load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_cache_key(question):
    # convert question to vector
    vector = model.encode(question).tolist()
    
    # check all existing cached questions
    cached_keys = r.keys("cache:*")
    
    best_match = None
    best_score = 0.0
    
    for key in cached_keys:
        data = json.loads(r.get(key))
        cached_vector = data["vector"]
        
        # calculate similarity between question and cached question
        score = np.dot(vector, cached_vector) / (
            np.linalg.norm(vector) * np.linalg.norm(cached_vector)
        )
        
        if score > best_score:
            best_score = score
            best_match = key

    # if similarity is above 0.95 it's close enough to return cached answer
    if best_score > 0.95:
        data = json.loads(r.get(best_match))
        print(f"\n--- CACHE HIT (similarity: {best_score:.2f}) ---")
        return data["answer"]
    
    return None  # no cache hit

def save_to_cache(question, answer):
    vector = model.encode(question).tolist()
    
    # create a unique key for this question
    key = "cache:" + hashlib.md5(question.encode()).hexdigest()
    
    # save question vector and answer together
    r.set(key, json.dumps({
        "question": question,
        "vector": vector,
        "answer": answer
    }))
    print("--- SAVED TO CACHE ---")