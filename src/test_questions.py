"""
Test different question formulations to find the best query
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

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(
    model="orca-mini",
    temperature=0.3,
    num_predict=200
)

def format_docs(docs):
    return "\n\n".join(f"[Doc {i}]: {doc.page_content[:500]}" for i, doc in enumerate(docs, 1))

system_prompt = (
    "You are a Financial Analyst answering questions based on company financial documents.\n"
    "Answer the question ONLY using the provided context.\n"
    "Be specific and cite details from the documents.\n"
    "If the exact information is not found in the context, clearly state: 'This information is not available in the provided documents.'\n"
    "Do NOT provide general knowledge or assumptions.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the RAG chain ONCE, not in the loop
rag_chain = (
    {"context": lambda x: format_docs(retriever.invoke(x["input"])), "input": lambda x: x["input"]}
    | prompt
    | llm
    | StrOutputParser()
)

# Test with different questions
test_questions = [
    "What are the main revenue streams?",
    "What are the three business segments?",
    "Tell me about Azure, Office, and Gaming revenue",
    "What are the operating segments?",
    "List the main business lines and their revenue",
    "What products generate revenue for Microsoft?",
    "Describe the business segments",
]

print("=" * 80)
print("TESTING QUESTION FORMULATIONS")
print("=" * 80)

for q in test_questions:
    print(f"\n\nQUESTION: {q}")
    print("-" * 80)
    
    docs = retriever.invoke(q)
    formatted = format_docs(docs)
    
    print(f"Retrieved context (first 400 chars):")
    print(formatted[:400] + "...\n")
    
    response = rag_chain.invoke({"input": q})
    print(f"Answer:\n{response}")
