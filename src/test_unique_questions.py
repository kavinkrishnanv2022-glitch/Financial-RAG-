"""
Test unique, specific financial questions to evaluate RAG performance
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = OllamaLLM(
    model="orca-mini",
    temperature=0.3,
    num_predict=300
)

def format_docs(docs):
    return "\n\n".join(f"[Doc {i}]: {doc.page_content[:500]}" for i, doc in enumerate(docs, 1))

system_prompt = (
    "You are a Financial Analyst answering questions based on company financial documents.\n"
    "Answer the question ONLY using the provided context.\n"
    "Be specific and cite details from the documents.\n"
    "If the exact information is not found in the context, clearly state: 'This information is not available in the provided documents.'\n"
    "If you have partial information (like components of a calculation), provide what you found and note what's missing.\n"
    "Do NOT provide general knowledge or assumptions.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the RAG chain once
rag_chain = (
    {"context": lambda x: format_docs(retriever.invoke(x["input"])), "input": lambda x: x["input"]}
    | prompt
    | llm
    | StrOutputParser()
)

# Unique, specific test questions
test_questions = [
    # Specific financial metrics
    "What was the gross margin percentage for the most recent fiscal year?",
    "How much revenue did the company generate from international markets?",
    "What is the company's total debt and what are the terms?",
    
    # Risk and challenges
    "What are the main cybersecurity and data privacy risks mentioned?",
    "What regulatory compliance challenges does the company face?",
    "How is the company affected by supply chain disruptions?",
    
    # Growth and strategy
    "What new product launches or initiatives did the company announce?",
    "Which geographic regions show the highest growth potential?",
    "How is the company investing in artificial intelligence and machine learning?",
    
    # Specific segments
    "What percentage of revenue comes from cloud services?",
    "Which business segment had the highest profit margin?",
    "What is the performance of the enterprise segment compared to consumer segment?",
    
    # Comparative questions
    "How did revenue change year-over-year in the most recent quarter?",
    "What was the trend in operating expenses over the past three years?",
    "How does the company's growth rate compare to the previous year?",
    
    # Research and development
    "How much does the company spend on research and development?",
    "What are the company's main innovation focus areas?",
    "How many engineers or R&D personnel does the company employ?",
    
    # Detailed segment questions
    "Describe the productivity and business processes segment in detail",
    "What are the key drivers of the intelligent cloud segment?",
    "How is the personal computing segment performing?",
    
    # Strategic partnerships and acquisitions
    "Has the company made any recent acquisitions? If so, what were they?",
    "What strategic partnerships or collaborations were announced?",
    "Are there any joint ventures mentioned in the documents?",
    
    # Specific product revenue
    "What is the revenue breakdown for Office 365, Microsoft Teams, and Dynamics products?",
    "How much revenue does Azure generate and what is its growth rate?",
    "What are the Windows licensing revenue trends?",
    
    # Financial metrics and ratios
    "What is the return on equity (ROE) for the company?",
    "What is the company's current ratio and debt-to-equity ratio?",
    "How has the company's cash flow from operations changed?",
]

print("=" * 80)
print("TESTING UNIQUE FINANCIAL QUESTIONS")
print("=" * 80)

for i, q in enumerate(test_questions, 1):
    print(f"\n\n[TEST {i}/{len(test_questions)}] QUESTION: {q}")
    print("-" * 80)
    
    docs = retriever.invoke(q)
    formatted = format_docs(docs)
    
    print(f"Retrieved context (first 400 chars):")
    print(formatted[:400] + "...\n")
    
    response = rag_chain.invoke({"input": q})
    print(f"Answer:\n{response}")
    print("-" * 80)
