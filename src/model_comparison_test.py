"""
Model Comparison Test - Tests different Ollama models side-by-side
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

print("=" * 80)
print("🧪 MODEL COMPARISON TEST")
print("=" * 80)

# Initialize components once
print("\n⏳ Loading components...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("✅ Components loaded!\n")

# Test question
test_question = "What are the main revenue streams?"

# Get retrieved documents
print(f"📄 Question: {test_question}")
print("-" * 80)
docs = retriever.invoke(test_question)
print(f"Retrieved {len(docs)} documents:")
for i, doc in enumerate(docs, 1):
    preview = doc.page_content[:150].replace('\n', ' ')
    print(f"  Doc {i}: ...{preview}...")

context = "\n\n".join(doc.page_content[:300] for doc in docs)

print("\n" + "=" * 80)
print("TESTING MODELS")
print("=" * 80)

# System prompt focused on synthesis (not verbatim copying)
system_prompt = (
    "You are a financial analyst.\n"
    "ANSWER DIRECTLY based on the context provided.\n"
    "Synthesize information - do NOT copy the context verbatim.\n"
    "If information is missing, say: 'Not found in document.'\n\n"
    "Context:\n{context}\n\n"
    "Provide a clear, direct answer:"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Models to test
models_to_test = ["mistral", "orca-mini", "neural-chat"]

for model in models_to_test:
    print(f"\n\n{'='*80}")
    print(f"Testing: {model}")
    print(f"{'='*80}\n")
    
    try:
        llm = OllamaLLM(
            model=model,
            temperature=0.3,
            num_predict=200
        )
        
        print(f"⏳ Sending query to {model}...")
        response = prompt.format(context=context, input=test_question)
        
        # Just test the LLM directly
        result = llm.invoke(test_question)
        
        print(f"\n✅ Response from {model}:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        
        # Analyze response
        if len(result) > 500:
            print(f"\n⚠️  Long response ({len(result)} chars) - might be echoing context")
        elif "not found" in result.lower():
            print(f"\n❓ Model claims info not found")
        elif result.startswith(context[:50]):
            print(f"\n❌ PROBLEM: Response starts with context verbatim")
        else:
            print(f"\n✅ Response appears to be synthesized")
            
    except Exception as e:
        print(f"❌ Error with {model}: {e}")
        print(f"   (Model might not be downloaded - try: ollama pull {model})")

print("\n" + "=" * 80)
print("✅ Comparison complete!")
print("=" * 80)
