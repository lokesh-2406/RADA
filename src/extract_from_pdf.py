#Load documents or pdfs
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader 
from langchain_community.document_loaders import TextLoader
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
    return documents