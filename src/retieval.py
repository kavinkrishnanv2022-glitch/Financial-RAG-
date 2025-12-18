import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Load Secrets
load_dotenv()
INDEX_NAME = "financial-rag"

def test_retrieval_only():
    print("🔎 Testing Retrieval (No LLM Cost)...")

    # 2. Local Embeddings (Matches ingest.py)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Connect to Pinecone
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )
    
    # 4. Run a Search
    query = "What are the risk factors regarding international trade?"
    print(f"\n❓ Question: {query}")
    
    # We ask for the top 3 matches
    docs = vectorstore.similarity_search(query, k=3)
    
    print(f"\n✅ Found {len(docs)} relevant documents in Pinecone!")
    
    print("\n--- Document 1 ---")
    print(docs[0].page_content[:300] + "...") # Print first 300 chars
    
    print("\n--- Document 2 ---")
    print(docs[1].page_content[:300] + "...")

if __name__ == "__main__":
    test_retrieval_only()