#streamlit run rag_app.py

import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

# --- Streamlit Config ---
st.set_page_config(page_title="📊 Financial Report RAG Assistant", layout="wide")

# --- Logo ---
#st.markdown(
#    """
#    <div style='text-align: center; margin-bottom: 20px;'>
#        <img src='https://trevpery.com/logo.png' width='200'/>
#    </div>
#    """,
#    unsafe_allow_html=True
# )

st.title("📄 Financial Report RAG Assistant")
st.markdown("Upload one or more financial documents and ask follow-up questions intelligently!")

# --- Check API Key ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OPENAI_API_KEY not found in Streamlit secrets.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# --- Session State Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# --- File Upload Section ---
uploaded_files = st.file_uploader("Upload your financial documents (PDFs)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("📚 Processing PDFs..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

        # Text Splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)

        # Embedding and Vector DB
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
        vectordb = FAISS.from_documents(chunks, embeddings)
        st.session_state.vectordb = vectordb

        st.success("✅ Documents processed and indexed!")

# --- Chat Input ---
if st.session_state.vectordb:
    st.subheader("🔍 Ask a Question")
    user_query = st.chat_input("Ask a question about the documents...")

    if user_query:
        with st.spinner("🤖 Thinking..."):
            llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vectordb.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
            )
            result = qa_chain(user_query)
            st.session_state.chat_history.append({"user": user_query, "bot": result['result']})

# --- Display Chat History ---
if st.session_state.chat_history:
    for exchange in st.session_state.chat_history:
        st.chat_message("user").write(exchange["user"])
        st.chat_message("assistant").write(exchange["bot"])

# --- Summarization ---
if st.session_state.vectordb:
    st.subheader("📌 Generate Summary (Optional)")

    if st.button("📝 Generate Summary"):
        with st.spinner("Summarizing the document..."):
            llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
            # We need chunks again for summarization
            # This assumes you still have the latest chunks in session — if not, store them too
            summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = summary_chain.run(st.session_state.vectordb.similarity_search("summary", k=50))
            st.success("✅ Summary generated")
            st.write(summary)
