"""
Model Response Quality Test Script
Tests whether the model synthesizes answers or just echoes context
"""
import os
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
print("🧪 MODEL RESPONSE QUALITY TEST")
print("=" * 80)

# Initialize components
print("\n⏳ Loading components...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Test with neural-chat model
llm = OllamaLLM(
    model="neural-chat",
    temperature=0.3,
    num_predict=200
)
print("✅ Components loaded!\n")

# Create RAG chain
def format_docs(docs):
    formatted = "\n\n---\n\n".join(doc.page_content[:300] for doc in docs)
    return formatted

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

rag_chain = (
    {"context": lambda x: format_docs(retriever.invoke(x["input"])), "input": lambda x: x["input"]}
    | prompt
    | llm
    | StrOutputParser()
)

# Test questions
test_questions = [
    "What are the main revenue streams for Microsoft?",
    "What is the total debt of the company?",
    "What are the geographic regions where the company operates?",
    "How many employees does the company have?",
    "What products does the company sell?"
]

print("=" * 80)
print("TESTING MODEL RESPONSES")
print("=" * 80)

for i, question in enumerate(test_questions, 1):
    print(f"\n[TEST {i}] Question: {question}")
    print("-" * 80)
    
    try:
        # Get retrieval context
        docs = retriever.invoke(question)
        print(f"📄 Retrieved {len(docs)} documents:")
        for j, doc in enumerate(docs, 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   Doc {j}: ...{preview}...")
        
        # Get model response
        print(f"\n🤖 Model Response:")
        response = rag_chain.invoke({"input": question})
        print(response)
        
        # Check if response looks like synthesis or verbatim context
        if len(response) > 300:
            print("\n⚠️  WARNING: Response is very long (>300 chars) - might be verbatim context")
        elif "not found" in response.lower() or "don't" in response.lower():
            print("\n✅ Model acknowledged missing information")
        elif response.startswith("The ") or response.startswith("It ") or response.startswith("Microsoft"):
            print("\n✅ Model provided direct answer (good synthesis)")
        else:
            print("\n❓ Unclear response format")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)

print("\n✅ Test complete!")
