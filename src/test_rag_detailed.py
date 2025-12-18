"""
Detailed RAG testing to see exactly what's happening at each step
"""
import os
import sys
# Fix encoding for Windows PowerShell
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
INDEX_NAME = "financial-rag"

print("=" * 80)
print("🔍 DETAILED RAG DIAGNOSTIC TEST")
print("=" * 80)

# Initialize components
print("\n⏳ Loading components...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Test with orca-mini first (simpler model)
llm = OllamaLLM(
    model="orca-mini",
    temperature=0.3,
    num_predict=200
)
print("✅ Components loaded!\n")

# Test question
test_question = "What are the main revenue streams?"

print("=" * 80)
print(f"TEST QUESTION: {test_question}")
print("=" * 80)

# Step 1: Retrieve documents
print("\n📄 STEP 1: Retrieving documents...")
docs = retriever.invoke(test_question)
print(f"✅ Retrieved {len(docs)} documents\n")

for i, doc in enumerate(docs, 1):
    print(f"--- Document {i} ---")
    print(f"Content preview (first 200 chars):\n{doc.page_content[:200]}\n")

# Step 2: Format documents
print("\n" + "=" * 80)
print("📝 STEP 2: Formatting context...")
formatted_context = "\n\n".join(doc.page_content[:300] for doc in docs)
print(f"Formatted context ({len(formatted_context)} chars):")
print(formatted_context[:500] + "...\n")

# Step 3: Create prompt
print("\n" + "=" * 80)
print("🎯 STEP 3: Creating prompt...")
system_prompt = (
    "You are answering a question based on financial document context.\n"
    "ANSWER THE QUESTION DIRECTLY using information from the context.\n"
    "Do not repeat the context verbatim.\n"
    "Provide a clear, concise answer.\n"
    "If the answer is not in the context, say: 'Not found in document.'\n\n"
    "Context:\n{context}\n\n"
    "Now answer the question:"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Get the full prompt
full_prompt = prompt.format(context=formatted_context, input=test_question)
print("Full prompt being sent to model:")
print("-" * 80)
print(full_prompt)
print("-" * 80)

# Step 4: Get model response
print("\n" + "=" * 80)
print("🤖 STEP 4: Sending to model (orca-mini)...")
response = llm.invoke(full_prompt)
print(f"✅ Model response:\n{response}\n")

# Step 5: Test with orca-mini via raw API call
print("\n" + "=" * 80)
print("📡 STEP 5: Testing raw API call to orca-mini...")
import json
import urllib.request

try:
    body = {
        "model": "orca-mini",
        "prompt": full_prompt,
        "stream": False,
        "temperature": 0.3,
        "num_predict": 200
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode())
        print(f"✅ API response:\n{result['response']}\n")
except Exception as e:
    print(f"❌ Error: {e}\n")

print("\n" + "=" * 80)
print("✅ DIAGNOSTIC TEST COMPLETE")
print("=" * 80)
