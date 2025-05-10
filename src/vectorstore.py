import os
from chromadb import Client
from chromadb.config import Settings
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
# 1.1 Initialize ChromaDB client and collection
def get_chroma_collection(persist_directory: str = "./chroma_db", collection_name: str = "pdf_chunks"):
    """
    Returns a ChromaDB collection, creating it if needed.
    """
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.makedirs(persist_directory, exist_ok=True)
    client = Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory
    ))
    embedding_fn = OpenAIEmbeddings(model="text-embedding-3-large")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    return collection

# 1.2 Create or update the vectorstore
def create_vectorstore(chunks: list[str], metadatas: list[dict], ids: list[str]):
    """
    Upserts the provided chunks into ChromaDB with embeddings and metadata.
    
    Args:
      - chunks: List of text chunks.
      - metadatas: Parallel list of metadata dicts (e.g. {"source": ..., "page": ...}).
      - ids: Unique IDs for each chunk.
    """
    collection = get_chroma_collection()
    collection.upsert(
        ids=ids,
        embeddings=None,        # letting Chroma call the embedding function internally
        documents=chunks,
        metadatas=metadatas
    )
    collection.persist()
    return collection

# 1.3 Inspecting the collection
def inspect_vectorstore(collection=None):
    """
    Iterates over all stored chunks and reports:
      - Total count
      - Any empty documents
      - Missing metadata fields
    """
    collection = collection or get_chroma_collection()
    # Retrieve everything
    res = collection.get(
        include=["ids", "documents", "metadatas"]
    )
    ids       = res["ids"]
    docs      = res["documents"]
    metadatas = res["metadatas"]

    report = {
        "total_chunks": len(ids),
        "empty_chunks": [],
        "missing_metadata": []
    }

    for idx, (doc, meta) in enumerate(zip(docs, metadatas)):
        if not doc.strip():
            report["empty_chunks"].append(ids[idx])
        # Example check: ensure each meta has a 'source' key
        if "source" not in meta:
            report["missing_metadata"].append(ids[idx])

    return report
