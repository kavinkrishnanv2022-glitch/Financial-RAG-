"""
Simple Single Model Test - Test one model at a time
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
print("🧪 SINGLE MODEL TEST")
print("=" * 80)

# Test with orca-mini first (we know it's available)
MODEL_TO_TEST = "orca-mini"

print(f"\nTesting model: {MODEL_TO_TEST}")
print("-" * 80)

try:
    # Initialize components
    print("⏳ Loading components...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Create LLM
    llm = OllamaLLM(
        model=MODEL_TO_TEST,
        temperature=0.3,
        num_predict=150
    )
    print(f"✅ Connected to Ollama ({MODEL_TO_TEST})\n")
    
    # Test question
    question = "What are the main revenue streams?"
    print(f"Question: {question}")
    print("-" * 80)
    
    # Retrieve documents
    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} documents")
    
    # Show context
    context_text = "\n\n".join(doc.page_content[:200] for doc in docs)
    print(f"\n📄 CONTEXT PROVIDED TO MODEL:")
    print(context_text)
    print("-" * 80)
    
    # Create simple prompt
    system_prompt = (
        "You are a financial analyst answering based on given context.\n"
        "Use ONLY the context below. Synthesize information - don't copy verbatim.\n"
        "If information isn't in context, say 'Not in context.'\n\n"
        "CONTEXT:\n{context}\n\n"
        "Now answer the question directly and concisely:"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create chain
    rag_chain = (
        {
            "context": lambda x: "\n\n".join(doc.page_content[:200] for doc in retriever.invoke(x["input"])),
            "input": lambda x: x["input"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Get response
    print(f"\n🤖 GETTING RESPONSE FROM {MODEL_TO_TEST}...\n")
    response = rag_chain.invoke({"input": question})
    
    print(f"RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
    # Analysis
    print(f"\n📊 ANALYSIS:")
    print(f"  - Response length: {len(response)} characters")
    print(f"  - Starts with 'The': {response.startswith('The')}")
    print(f"  - Says 'not found': {'not found' in response.lower()}")
    print(f"  - Is direct answer: {not response.startswith(context_text[:50])}")
    
    if len(response) < 200 and not "not found" in response.lower():
        print(f"\n✅ GOOD: Response is concise and direct!")
    elif "not found" in response.lower():
        print(f"\n⚠️  Model claims information not in context")
    elif len(response) > 500:
        print(f"\n❌ PROBLEM: Response is too long - may be echoing context")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
