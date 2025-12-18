"""
Terminal-based RAG Q&A Test
Ask questions and get answers directly in the terminal
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

print("=" * 70)
print("📊 FINANCIAL RAG - TERMINAL Q&A TEST")
print("=" * 70)

# Initialize components
print("\n⏳ Loading components...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = OllamaLLM(model="orca-mini", temperature=0.3, num_predict=300)
print("✅ All components loaded!\n")

# Create RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

rag_chain = (
    {"context": lambda x: format_docs(retriever.invoke(x["input"])), "input": lambda x: x["input"]}
    | prompt
    | llm
    | StrOutputParser()
)

# Interactive loop
print("Type 'quit' or 'exit' to end\n")
print("-" * 70)

while True:
    question = input("\n❓ Ask a question: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if not question:
        print("⚠️  Please enter a question")
        continue
    
    print(f"\n⏳ Processing: '{question}'")
    print("🔍 Retrieving documents...")
    
    try:
        print("💭 Generating response...\n")
        response = rag_chain.invoke({"input": question})
        
        print("=" * 70)
        print("📝 ANSWER:")
        print("=" * 70)
        print(response)
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
