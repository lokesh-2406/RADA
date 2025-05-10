#Load documents or pdfs
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader 
from langchain_community.document_loaders import TextLoader
from src.chunking.chunking import character_chunk_documents
from src.vectorstore import create_vectorstore
import tqdm
import os
import fitz  # PyMuPDF
from loguru import logger
def load_documents(pdfs):
    '''Load documents from a directory.'''
    # Load documents from a directory
    all_chunks = []
    all_metadatas = []
    logger.info(f"fitz version: {fitz.__version__}")
    logger.info(f"PyMuPDF version: {fitz.__doc__}")
    for pdf_path in pdfs:
        # 1. Load with PyMuPDF
        loader = PyMuPDFLoader(pdf_path,)
        documents = loader.load()
        
        # 2. Extract native metadata
        doc = fitz.open(pdf_path)
        base_metadata = {
            "source": os.path.basename(pdf_path),
            "author": doc.metadata.get("author"),
            "title": doc.metadata.get("title"),
            "creation_date": doc.metadata.get("creationDate")
        }
        
        # 3. Add page-specific metadata
        for i, page_doc in enumerate(documents):
            page_metadata = {
                **base_metadata,
                "page": i+1,
                "section": page_doc.metadata.get("section", "")
            }
            all_metadatas.append(page_metadata)
        
        # 4. Chunk with metadata
        chunks = character_chunk_documents(documents)
        all_chunks.extend(chunks)
    
    # 5. Generate unique IDs
    ids = [f"{meta['source']}_{meta['page']}_{idx}" for idx, meta in enumerate(all_metadatas)]
    logger.info(f"Loaded {len(documents)} documents and split them into {len(all_chunks)} chunks, using CharacterTextSplitter.")
    # 6. Index
    collection = create_vectorstore(all_chunks, all_metadatas, ids)
    logger.info(f"Indexed {len(chunks)} chunks into ChromaDB.")

    # Return raw docs *and* the Chroma collection object
    return documents, collection