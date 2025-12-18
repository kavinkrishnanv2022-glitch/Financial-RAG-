import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # <-- NEW LIBRARY
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "financial-rag"

def ingest_data():
    print("🚀 Starting LOCAL Ingestion Pipeline...")

    # 1. Load PDF
    pdf_path = "C:/Users/kavin/Downloads/financial.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at {pdf_path}")
        return

    print("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    print(f"   - Loaded {len(raw_docs)} pages.")

    # 2. Split Text
    print("✂️  Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(raw_docs)
    print(f"   - Created {len(documents)} chunks to process.")

    # 3. Initialize Local Embeddings (The Fix)
    print("💻 Initializing Local HuggingFace Model (all-MiniLM-L6-v2)...")
    # This runs LOCALLY on your CPU. No API calls. No Rate Limits.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Upload to Pinecone
    print(f"📡 Uploading to Pinecone index '{INDEX_NAME}'...")
    
    # We can batch huge amounts now because we aren't limited by Google
    batch_size = 100 
    total_docs = len(documents)

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        print(f"   - Processing batch {i//batch_size + 1} ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)

    print("✅ Success! All data is indexed without rate limits.")

if __name__ == "__main__":
    ingest_data()