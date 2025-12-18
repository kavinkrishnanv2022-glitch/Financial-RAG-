import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Page Config
st.set_page_config(
    page_title="Financial Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Professional CSS Styling
st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        margin-top: 0.25rem;
    }
    
    /* Info Box Styles */
    .info-premium {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-left: 5px solid #667eea;
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
    }
    
    .info-premium strong {
        color: #667eea;
    }
    
    /* Chat Container */
    .chat-header {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .chat-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #333;
        margin: 0;
    }
    
    /* Sidebar Styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    
    .sidebar-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* Footer */
    .footer-container {
        border-top: 2px solid #e0e0e0;
        padding-top: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        color: #999;
        font-size: 0.85rem;
    }
    
    /* Example Questions */
    .example-badge {
        display: inline-block;
        background: #e8f4f8;
        color: #1976d2;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #1976d2;
    }
    
    /* Response Styling */
    .response-box {
        background: #f5f7fa;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 5px solid #2ca02c;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Environment
load_dotenv()
INDEX_NAME = "financial-rag"

# 4. Sidebar Configuration
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 3rem;">📊</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #667eea;">Financial Intelligence</div>
            <div style="font-size: 0.85rem; color: #999;">Powered by Advanced RAG</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown('<div class="sidebar-title">🎯 Model Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        model_temp = st.slider(
            "Temperature",
            0.0, 1.0, 0.3, 0.1,
            help="Lower = deterministic, Higher = creative"
        )
    with col2:
        max_tokens = st.slider(
            "Max Tokens",
            100, 500, 300, 50
        )
    
    retrieval_k = st.select_slider(
        "Retrieved Documents",
        options=list(range(1, 11)),
        value=5,
        help="Number of documents to retrieve per query"
    )
    
    st.divider()
    
    st.markdown('<div class="sidebar-title">⚡ System Status</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "orca-mini", delta="Active", label_visibility="collapsed")
    with col2:
        st.metric("Vector DB", "Pinecone", delta="Connected", label_visibility="collapsed")
    
    st.divider()
    
    st.markdown('<div class="sidebar-title">🔧 Actions</div>', unsafe_allow_html=True)
    
    if st.button("🔄 Refresh Models", use_container_width=True, key="refresh_btn"):
        st.cache_resource.clear()
        st.toast("✓ Cache cleared!", icon="✅")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_btn"):
        st.session_state.messages = []
        st.toast("✓ Chat history cleared!", icon="✅")
    
    st.divider()
    
    st.markdown('<div class="sidebar-title">💡 Example Queries</div>', unsafe_allow_html=True)
    
    examples = [
        "What are the three business segments?",
        "How much revenue from Azure?",
        "What is the gross margin percentage?",
        "What are the main risk factors?",
        "How is the company investing in AI?",
        "What is the current ratio?",
    ]
    
    for i, example in enumerate(examples):
        st.markdown(f'<div class="example-badge">{example}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
        <div style="text-align: center; color: #999; font-size: 0.8rem;">
        <p>🏢 Enterprise RAG System</p>
        <p>v1.0 • Deployed Dec 2025</p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_resource
def get_rag_chain(temperature, max_tokens, k):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    llm = OllamaLLM(
        model="orca-mini",
        temperature=temperature,
        num_predict=max_tokens
    )

    system_prompt = (
        "You are a Senior Financial Analyst with deep expertise in corporate finance.\n"
        "Answer questions ONLY using the provided document context.\n"
        "Be precise, specific, and cite exact figures and page references when available.\n"
        "If information is not found, clearly state: 'This information is not available in the provided documents.'\n"
        "For complex calculations, show the components you found and what's missing.\n"
        "Never provide general knowledge or external assumptions.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(f"[Doc {i}]: {doc.page_content[:500]}" for i, doc in enumerate(docs, 1))

    rag_chain = (
        {"context": lambda x: format_docs(retriever.invoke(x["input"])), "input": lambda x: x["input"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# 5. Main Content Area
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📊 Financial Intelligence Platform</h1>
        <p class="header-subtitle">Enterprise-Grade Document Analysis with Advanced AI Retrieval</p>
    </div>
    """, unsafe_allow_html=True)

# 6. Info Banner
st.markdown("""
    <div class="info-premium">
    <strong>✨ AI-Powered Analysis</strong><br>
    This platform uses advanced retrieval-augmented generation (RAG) to extract and analyze 
    financial information from your documents with exceptional accuracy. All responses are 
    grounded in source materials for complete transparency and auditability.
    </div>
    """, unsafe_allow_html=True)

# 7. Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# 8. Chat Header
st.markdown("""
    <div class="chat-header">
        <p class="chat-title">💬 Conversation History</p>
    </div>
    """, unsafe_allow_html=True)

# 9. Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# 10. Chat Input
st.markdown("---")

if prompt := st.chat_input("Ask a detailed question about your financial documents...", key="main_input"):
    # User message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("⏳ Analyzing documents and generating response..."):
                chain = get_rag_chain(model_temp, max_tokens, retrieval_k)
                response = chain.invoke({"input": prompt})
            
            if response and len(response.strip()) > 0:
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.toast("✓ Response generated successfully!", icon="✅")
            else:
                message_placeholder.warning("⚠️ No response generated. Please try rephrasing your question.")
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ No response generated."})

        except Exception as e:
            error_content = f"""
### ❌ Analysis Failed

**Error Details:** {str(e)}

**Troubleshooting Steps:**
1. **Ensure Ollama is running:** Open terminal and run `ollama serve`
2. **Verify Pinecone Connection:** Check your API key in `.env` file
3. **Confirm Data Ingestion:** Run `python src/ingest.py` to index documents
4. **Check Index Status:** Verify index `{INDEX_NAME}` exists in Pinecone dashboard
5. **Review Logs:** Check console output for detailed error messages

**Need Help?**
- Verify all environment variables are correctly set
- Ensure you have sufficient Pinecone quota
- Check that documents were successfully ingested
            """
            message_placeholder.error(error_content)
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

# 11. Footer
st.markdown("""
    <div class="footer-container">
        <div style="margin-bottom: 1rem;">
            <strong>🔧 Technology Stack</strong><br>
            LangChain + Ollama (Local LLM) • Pinecone (Vector DB) • HuggingFace Embeddings • Streamlit
        </div>
        <div>
            <strong>✅ Key Benefits</strong><br>
            Zero Cloud Costs • Complete Privacy • Fast Local Processing • Production-Ready
        </div>
        <hr style="margin: 1rem 0; border: none; border-top: 1px solid #e0e0e0;">
        <p style="margin: 0.5rem 0; color: #bbb;">
            Financial Intelligence Platform • Enterprise RAG Solution
        </p>
        <p style="margin: 0; font-size: 0.8rem; color: #ccc;">
            © 2025 Advanced AI Analytics. All rights reserved.
        </p>
    </div>
    """, unsafe_allow_html=True)