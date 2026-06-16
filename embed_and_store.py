import chromadb  # our vector database
from sentence_transformers import SentenceTransformer  # our embedding model
from load_pdf import load_pdf  # importing our function from the first file

# load the embedding model (downloads once, then cached on your laptop)
model = SentenceTransformer("all-MiniLM-L6-v2")

# create a local ChromaDB database (saves to a folder called "db" on your laptop)
client = chromadb.PersistentClient(path="db")

# create a collection (like a table in a normal database, holds all our vectors)
collection = client.get_or_create_collection(name="pdf_chunks")

def embed_and_store(pdf_path):
    chunks = load_pdf(pdf_path)  # get all chunks from the first file
    
    texts = [c["text"] for c in chunks]   # extract just the text from each chunk
    pages = [str(c["page"]) for c in chunks]  # extract just the page numbers
    ids = [f"chunk_{i}" for i in range(len(chunks))]  # give each chunk a unique id
    
    print("Embedding chunks... this may take a minute...")
    embeddings = model.encode(texts).tolist()  # convert all chunks to vectors
    
    # store everything in ChromaDB
    collection.add(
        ids=ids,                                      # unique ids
        embeddings=embeddings,                        # the vectors
        documents=texts,                              # the actual text
        metadatas=[{"page": p} for p in pages]       # the page numbers
    )
    
    print(f"Stored {len(chunks)} chunks in the database!")

# run it — change "your_file.pdf" to your actual pdf filename
embed_and_store("paper.pdf")
