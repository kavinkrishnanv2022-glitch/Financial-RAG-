"""
Debug: See exactly what prompt is being sent and what the model returns
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

load_dotenv()
INDEX_NAME = "financial-rag"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Just 1 doc to keep it simple

llm = OllamaLLM(
    model="orca-mini",
    temperature=0.3,
    num_predict=200
)

question = "Describe the three operating business segments"
print(f"Question: {question}\n")

# Get retrieved docs
docs = retriever.invoke(question)
print(f"Retrieved {len(docs)} document(s):")
print(f"Content:\n{docs[0].page_content[:300]}\n")
print("=" * 80)

# Format context
context = f"[DOC]: {docs[0].page_content[:400]}"
print(f"Formatted context for model:\n{context}\n")
print("=" * 80)

# Create the prompt
system_prompt = (
    "You are a helpful assistant answering questions based on financial documents.\n"
    "Answer the question directly using the provided context.\n"
    "If the answer is not found, say 'Not found.'\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {input}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
])

# Get the full formatted prompt
full_prompt_text = system_prompt.format(context=context, input=question)
print(f"Full prompt being sent to model:\n")
print(full_prompt_text)
print("\n" + "=" * 80)
print("MODEL RESPONSE:\n")

# Call model directly
import requests
body = {
    "model": "orca-mini",
    "prompt": full_prompt_text,
    "stream": False,
    "temperature": 0.3,
    "num_predict": 200
}

response = requests.post("http://localhost:11434/api/generate", json=body, timeout=60)
result = response.json()
print(result["response"])
