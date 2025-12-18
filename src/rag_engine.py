import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load Secrets
load_dotenv()
INDEX_NAME = "financial-rag"

def setup_rag_chain():
    print("🧠 Loading RAG models...")

    # 2. Local Embeddings (Must match ingest.py)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Connect to Pinecone
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 4. Connect to Local Ollama LLM (no API costs!)
    # Fast, lightweight model - great for RAG tasks
    llm = OllamaLLM(
        model="orca-mini",  # Super fast responses (3-5 seconds)
        temperature=0.3
    )

    # 5. Create the Prompt (Modern Format)
    system_prompt = (
        "You are a Senior Financial Analyst. "
        "Answer ONLY based on the provided context. "
        "Do not provide general knowledge answers. "
        "If the information is not in the context, say 'I cannot find this information in the document.' "
        "Be specific and cite details from the context."
        "\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Build the Modern Chain using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Extract just the input string for the retriever
    rag_chain = (
        {"context": lambda x: format_docs(retriever.invoke(x["input"])), "input": lambda x: x["input"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Quick Test Block ---
if __name__ == "__main__":
    chain = setup_rag_chain()
    
    query = "What are the risk factors regarding international trade?"
    print(f"\n❓ Question: {query}\n")
    
    response = chain.invoke({"input": query})
    
    print("🤖 Answer:")
    print(response)