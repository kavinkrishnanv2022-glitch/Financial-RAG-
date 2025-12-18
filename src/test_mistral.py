"""
Final test: Mistral model with multiple questions
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
print("TESTING WITH MISTRAL MODEL")
print("=" * 80)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = OllamaLLM(
    model="mistral",
    temperature=0.3,
    num_predict=150
)

def format_docs(docs):
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

# Test questions
test_questions = [
    "What are the three operating business segments?",
    "Tell me about Azure revenue",
    "What is Office revenue?",
]

for question in test_questions:
    print(f"\nQUESTION: {question}")
    print("-" * 80)
    response = rag_chain.invoke({"input": question})
    print(f"ANSWER: {response}\n")

print("=" * 80)
print("TEST COMPLETE!")
print("=" * 80)
