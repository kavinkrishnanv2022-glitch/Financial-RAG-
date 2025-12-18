"""
Simple test to check each component step-by-step
"""
import os
from dotenv import load_dotenv

print("=" * 60)
print("⚡ QUICK COMPONENT TEST")
print("=" * 60)

# Test 1: Ollama
print("\n1️⃣  Testing Ollama connection...")
try:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="orca-mini", temperature=0.3)
    
    print("   ⏳ Sending test query...")
    result = llm.invoke("Say 'Hello' in one word only")
    print(f"   ✅ Response: {result}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Embeddings
print("\n2️⃣  Testing HuggingFace Embeddings...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    query = "revenue streams"
    embedding = embeddings.embed_query(query)
    print(f"   ✅ Embedding generated (size: {len(embedding)})")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Pinecone
print("\n3️⃣  Testing Pinecone connection...")
try:
    load_dotenv()
    from langchain_pinecone import PineconeVectorStore
    
    vectorstore = PineconeVectorStore(
        index_name="financial-rag",
        embedding=embeddings
    )
    
    print("   ✅ Pinecone connected")
    
    # Try to retrieve
    print("   ⏳ Retrieving documents...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke("What is revenue?")
    
    if docs:
        print(f"   ✅ Retrieved {len(docs)} document(s)")
        print(f"   Content preview: {docs[0].page_content[:100]}...")
    else:
        print("   ⚠️  No documents retrieved")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Component test complete!")
print("=" * 60)
