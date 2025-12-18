"""
Debug script to test RAG pipeline step-by-step
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
INDEX_NAME = "financial-rag"

print("=" * 60)
print("🔍 DEBUGGING RAG PIPELINE")
print("=" * 60)

# Step 1: Initialize Embeddings
print("\n1️⃣  Loading HuggingFace Embeddings...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("   ✅ Embeddings loaded successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Step 2: Connect to Pinecone
print("\n2️⃣  Connecting to Pinecone...")
try:
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    print("   ✅ Pinecone connected successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Step 3: Test retrieval
print("\n3️⃣  Testing Document Retrieval...")
query = "What are the main revenue streams?"
try:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    print(f"   ✅ Retrieved {len(docs)} documents")
    
    if docs:
        print("\n   📄 RETRIEVED DOCUMENTS:")
        for i, doc in enumerate(docs, 1):
            print(f"\n   Document {i}:")
            print(f"   Content: {doc.page_content[:200]}...")
            if hasattr(doc, 'metadata'):
                print(f"   Metadata: {doc.metadata}")
    else:
        print("   ⚠️  No documents retrieved - check if data was ingested!")
except Exception as e:
    print(f"   ❌ Error retrieving documents: {e}")

# Step 4: Test LLM
print("\n4️⃣  Testing Ollama LLM...")
try:
    llm = OllamaLLM(
        model="orca-mini",
        temperature=0.3
    )
    print("   ✅ Ollama connected successfully")
    
    # Simple test
    print("\n   🧪 Testing simple inference...")
    response = llm.invoke("What is 2+2?")
    print(f"   Response: {response}")
    print("   ✅ LLM is working")
except Exception as e:
    print(f"   ❌ Error with LLM: {e}")

# Step 5: Test full RAG chain
print("\n5️⃣  Testing Full RAG Chain...")
try:
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    system_prompt = (
        "You are a Senior Financial Analyst. "
        "Use the given context to answer the question. "
        "If you don't know, say so."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    rag_chain = (
        {"context": lambda x: format_docs(retriever.invoke(x["input"])), "input": lambda x: x["input"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("   🧪 Asking: 'What are the main revenue streams?'")
    result = rag_chain.invoke({"input": "What are the main revenue streams?"})
    
    print("\n   📝 RESPONSE:")
    print("   " + "=" * 55)
    print(f"   {result}")
    print("   " + "=" * 55)
    print("\n   ✅ Full RAG chain working!")
    
except Exception as e:
    print(f"   ❌ Error with RAG chain: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Debug test complete!")
print("=" * 60)
