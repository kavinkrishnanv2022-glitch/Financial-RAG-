"""
Compare orca-mini vs mistral on the same context
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import requests
import json
import time

load_dotenv()
INDEX_NAME = "financial-rag"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

question = "Describe the three operating business segments"
docs = retriever.invoke(question)
context = f"[DOC]: {docs[0].page_content[:400]}"

system_prompt = (
    "You are a helpful assistant answering questions based on financial documents.\n"
    "Answer the question directly using the provided context.\n"
    "If the answer is not found, say 'Not found.'\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {input}"
)

full_prompt = system_prompt.format(context=context, input=question)

models_to_test = ["orca-mini", "mistral"]

for model in models_to_test:
    print(f"\n{'='*80}")
    print(f"Testing: {model}")
    print('='*80)
    
    body = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "temperature": 0.3,
        "num_predict": 200
    }
    
    try:
        print(f"Sending request... (this may take a while for {model})")
        start = time.time()
        response = requests.post("http://localhost:11434/api/generate", json=body, timeout=120)
        elapsed = time.time() - start
        result = response.json()
        print(f"Time: {elapsed:.1f}s")
        print(f"Response:\n{result['response']}")
    except Exception as e:
        print(f"Error: {e}")
