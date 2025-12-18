"""
Simple test: Check if a better question gets better results
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
    return "\n\n".join(f"[Doc {i}]: {doc.page_content[:400]}" for i, doc in enumerate(docs, 1))

system_prompt = (
    "You are answering based on financial document context.\n"
    "ANSWER DIRECTLY using the provided context.\n"
    "If information is not in the context, say 'Not found.'\n\n"
    "Context:\n{context}"
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

# Test a better formulated question
question = "Describe the three operating business segments and their main products"
print(f"Question: {question}\n")
response = rag_chain.invoke({"input": question})
print(f"Answer:\n{response}\n")
