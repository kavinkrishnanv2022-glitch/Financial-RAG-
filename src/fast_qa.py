"""
Fast Terminal RAG Q&A - Windows compatible
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from threading import Thread
import time

load_dotenv()
INDEX_NAME = "financial-rag"

print("=" * 70)
print("📊 FINANCIAL RAG - FAST Q&A")
print("=" * 70)

# Initialize components
print("\n⏳ Loading components...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Reduced from 3 to 2

# Use mistral - best quality, slower but worth it
llm = OllamaLLM(
    model="mistral",  # Best instruction following - worth the ~100s wait
    temperature=0.3,
    num_predict=150  # Reduced to speed up slightly
)
print("✅ All components loaded!\n")

# Create RAG chain with shorter context
def format_docs(docs):
    # Truncate each doc to save processing time
    return "\n\n".join(doc.page_content[:300] for doc in docs)

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

# Interactive loop
print("Type 'quit' to end.\n")
print("-" * 70)

while True:
    question = input("\n❓ Question: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if not question:
        print("⚠️  Enter a question")
        continue
    
    print(f"⏳ Processing...")
    
    try:
        response = rag_chain.invoke({"input": question})
        
        print("\n" + "=" * 70)
        print(response)
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
