"""
Direct REST API Test - Bypasses LangChain to test Ollama directly
"""
import requests
import json
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
INDEX_NAME = "financial-rag"

print("=" * 80)
print("🧪 DIRECT OLLAMA REST API TEST")
print("=" * 80)

# Get retrieved documents
print("\n1️⃣  Retrieving documents from Pinecone...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

question = "What are the main revenue streams?"
docs = retriever.invoke(question)

context = "\n\n".join(doc.page_content[:250] for doc in docs)
print(f"✅ Retrieved {len(docs)} documents\n")

print("📄 CONTEXT:")
print("-" * 80)
print(context)
print("-" * 80)

# Test each model
models = ["orca-mini", "neural-chat", "mistral"]

for model in models:
    print(f"\n\n{'='*80}")
    print(f"Testing: {model}")
    print(f"{'='*80}\n")
    
    try:
        # Create prompt
        full_prompt = f"""You are a financial analyst. Answer based ONLY on this context.
Synthesize the information - don't copy verbatim.
If not in context, say "Not found in document."

CONTEXT:
{context}

QUESTION: {question}

ANSWER (direct and concise):"""
        
        # Call Ollama
        print(f"⏳ Calling {model}...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "temperature": 0.3,
                "num_predict": 150
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            
            print(f"✅ Response from {model}:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            
            # Analysis
            if len(answer) > 400:
                print(f"⚠️  Long response ({len(answer)} chars)")
            elif "not found" in answer.lower():
                print(f"❌ Claims info not in context")
            elif len(answer) > 50:
                print(f"✅ Good synthesized answer")
            else:
                print(f"⚠️  Short response")
        else:
            print(f"❌ Error: Status {response.status_code}")
            
    except requests.Timeout:
        print(f"❌ TIMEOUT: {model} took too long (>120 seconds)")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ Test complete!")
print("=" * 80)
