from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader 
from langchain_community.document_loaders import TextLoader
from src.chunking.chunking import character_chunk_documents 
from src.chunking.semantic_chunking import semantic_chunk_documents
from src.vectorstore import create_vectorstore
import tqdm
import uuid
import os
import fitz  # PyMuPDF
from loguru import logger

def load_documents(pdfs):
    '''Load documents from uploaded PDFs and create a vector store.'''
    if not pdfs:
        logger.warning("No PDFs provided")
        return None
        
    all_chunks = []
    all_metadatas = []
    logger.info(f"fitz version: {fitz.__version__}")
    logger.info(f"PyMuPDF version: {fitz.__doc__}")
    
    for pdf_path in pdfs:
        logger.info(f"Processing PDF: {pdf_path}")
        # 1. Load with PyMuPDF
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            
            # 2. Extract native metadata
            doc = fitz.open(pdf_path)
            base_metadata = {
                "source": os.path.basename(pdf_path),
                "author": doc.metadata.get("author"),
                "title": doc.metadata.get("title"),
                "creation_date": doc.metadata.get("creationDate")
            }
            
            # 3. Process each page
            page_metadatas = []
            for i, page_doc in enumerate(documents):
                page_metadata = {
                    **base_metadata,
                    "page": i+1,
                    "section": page_doc.metadata.get("section", "")
                }
                page_metadatas.append(page_metadata)
            
            # 4. Chunk documents
            chunks = semantic_chunk_documents(documents)
            
            # Ensure chunks and metadata align
            for i, chunk in enumerate(chunks):
                # Find the corresponding page metadata
                page_num = chunk.metadata.get("page", 1)
                # Use page number to find the right metadata (adjust for 0-indexing if needed)
                chunk_metadata = next(
                    (meta for meta in page_metadatas if meta["page"] == page_num), 
                    page_metadatas[0] if page_metadatas else {"source": os.path.basename(pdf_path)}
                )
                all_chunks.append(chunk)
                all_metadatas.append(chunk_metadata)
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
    
    if not all_chunks:
        logger.warning("No chunks were created from the documents")
        return None
    
    # 5. Generate unique IDs
    ids = [str(uuid.uuid4()) for _ in all_chunks]
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(pdfs)} documents")
    
    # 6. Create vector store
    try:
        # Extract text from Documents
        texts = [doc.page_content for doc in all_chunks]
        # Extract metadata
        metadatas = [doc.metadata for doc in all_chunks]
        
        # Create the vector store
        collection = create_vectorstore(texts, metadatas, ids)
        logger.info(f"Successfully created vector store with {len(texts)} chunks")
        return collection
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None