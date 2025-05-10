#Load documents or pdfs
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader 
from langchain_community.document_loaders import TextLoader
from src.chunking.chunking import character_chunk_documents
from src.vectorstore import create_vectorstore
import tqdm
import os
from dotenv import load_dotenv
def load_documents(pdfs):
    '''Load documents from a directory.'''
    load_dotenv()
    #setup openai api key from environment variable
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    # Load documents from a directory
    loader = DirectoryLoader(
        pdfs,
        glob="**/*.pdf", #which tyoe of files to load
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    
    # 1. Chunk
    chunks = character_chunk_documents(documents)

    # 2. Prepare metadata & IDs
    pdf_name = os.path.basename(pdfs[0].name)  # assume single-dir
    ids = [f"{pdf_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_name} for _ in chunks]

    # 3. Index
    collection = create_vectorstore(chunks, metadatas, ids)

    # Return raw docs *and* the Chroma collection object
    return documents, collection