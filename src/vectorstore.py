from sentence_transformers import SentenceTransformer
import json
import os
import chromadb

# It converts any sentence into a 384-dimensional vector
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# This is where ChromaDB will save its data on disk
# So we don't have to re-embed 16,000 rows every time we run the app
CHROMA_PATH = "chroma_db"

# The name of the "table" inside ChromaDB where we store our medical data
COLLECTION_NAME = "medquad"


def load_model():

    """
    Loads the sentence transformer model.
    First run downloads it (~90MB). After that it's cached locally.
    """

    print(f"Loading embedding model {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Model Loaded")
    return model


def build_vector_store(data_path="data/processed/medquad_clean.json"):

    """
    Reads the cleaned dataset, embeds every Q&A pair,
    and stores them in ChromaDB on disk.
    
    We embed the QUESTION so that when a user describes symptoms,
    we can find the most similar medical questions and return their answers.
    """

    #Loading cleaned dataset
    with open(data_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} Q&A pairs")

    #Loading embedding model
    model = load_model()

    #Setting up ChromaDB

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={
            "hnsw:space" : "cosine"
        }
    )

    existing_count= collection.count()

    if existing_count > 0:
        print(f"Collection already has {existing_count} embeddings")
        return collection
    
    #Embed and Store in batches

    BATCH_SIZE = 500

    print(f"Embedding {len(data)} Q&A pairs in batches of size {BATCH_SIZE}")

    for i in range(0, len(data), BATCH_SIZE):

        batch = data[i: i+BATCH_SIZE]

        questions = [row['question'] for row in batch]

        answers = [row['answer'] for row in batch]

        embeddings = model.encode(questions, convert_to_tensor=False).tolist()

        ids= [str(i+j) for j in range(len(batch))]

        collection.add(
            embeddings = embeddings,
            documents = answers,  # gets returned when we search for a particular question
            metadatas = [{"question":q} for q in questions],
            ids = ids
        )

    print(f"{collection.count} embeddings stored in Vector DB")

    return collection


def get_collection():

    """
    Quick helper function used by other files (like vectorstore retrieval).
    Just opens the existing ChromaDB and returns the collection.
    No re-embedding — just loads what's already on disk.
    """

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata ={
            "hnsw:space" : "cosine" 
        }
    )

    return collection


def search(query: str, n_results: int=3):


    model = load_model()

    collection = get_collection()

    #converting user's query into a vector using the same model
    encoded_query = model.encode(query, convert_to_tensor=False).tolist()

    results = collection.query(
        query_embeddings= [encoded_query],
        n_results= n_results
    )

    retrieved=[]

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append(
            {
                "question" : meta['question'],
                "answer" : doc
            }
        )

    return retrieved    


if __name__=="__main__":

    build_vector_store()

    # Test a sample search to verify it's working
    print("\nTesting search...")
    test_query = "I have a headache and high fever for 2 days"
    results = search(test_query)

    print(f"\nQuery: '{test_query}'")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Matched Q: {r['question']}")
        print(f"  Answer preview: {r['answer']}...")
        print()










