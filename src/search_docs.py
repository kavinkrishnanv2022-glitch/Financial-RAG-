"""
Search Pinecone for documents that mention specific revenue streams
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
INDEX_NAME = "financial-rag"

print("=" * 80)
print("SEARCHING FOR REVENUE STREAM DOCUMENTS")
print("=" * 80)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Try different search queries
search_queries = [
    "Azure revenue",
    "Office revenue",
    "revenue by product segment",
    "Microsoft revenue breakdown",
    "business segments revenue",
]

for query in search_queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)
    docs = retriever.invoke(query)
    print(f"Found {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:300].replace('\n', ' ')
        print(f"Doc {i}: {preview}...")
        print()
