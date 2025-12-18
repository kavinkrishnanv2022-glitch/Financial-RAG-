import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Load the secrets
load_dotenv()

# 2. Get the keys
google_key = os.getenv("GOOGLE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

print(f"✅ Google Key Loaded: {'Yes' if google_key else 'No'}")
print(f"✅ Pinecone Key Loaded: {'Yes' if pinecone_key else 'No'}")

try:
    if pinecone_key:
        pc = Pinecone(api_key=pinecone_key)
        print("✅ Connection to Pinecone servers successful!")
    
    if google_key:
        # We use the 'embedding-001' model from Google
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Let's try to embed a simple word to ensure the API is actually working
        test_vector = embeddings.embed_query("finance")
        print(f"✅ Gemini Embeddings generating successfully! (Vector length: {len(test_vector)})")

except Exception as e:
    print(f"❌ Error: {e}")