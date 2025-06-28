from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def ingest_pdf(pdf_path):
    # Check if file really exists before loading
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} does not exist. Check your path!")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    vectordb.persist()
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")
    # Optionally show a sample chunk for debug:
    if chunks:
        print("Sample chunk:\n", chunks[0].page_content[:500])

if __name__ == "__main__":
    ingest_pdf("data/legal/nda_sample.pdf")