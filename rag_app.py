#streamlit run rag_app.py
import os
import tempfile
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain


# Load API Key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

if not openai_api_key:
    st.error("OPENAI_API_KEY not set. Please set it in .streamlit/secrets.toml.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# Streamlit UI
st.set_page_config(page_title="📊 Financial Report RAG Assistant", layout="wide")
st.title("📄 Financial Report RAG Assistant")
st.markdown("Upload a PDF report, ask questions, and get intelligent answers!")

# File upload
uploaded_file = st.file_uploader("Upload your Annual Report (PDF)", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Load & split
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Embeddings & Vector DB
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.persist()

        st.success("✅ PDF processed and indexed!")

        # Question interface
        st.subheader("🔍 Ask Questions")
        user_query = st.text_input("Type your question about the report")
        if user_query:
            with st.spinner("Answering..."):
                llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                )
                result = qa_chain(user_query)
                st.write("### ✅ Answer:")
                st.write(result['result'])

                with st.expander("🔎 Sources"):
                    for doc in result['source_documents']:
                        st.markdown(f"**Page Source:** {doc.metadata.get('page', 'Unknown')}")
                        st.write(doc.page_content[:500] + "...")
                        
        st.subheader("📌 Summary of Document (Optional)")

        if st.button("Generate Summary"):
           with st.spinner("Summarizing..."):
              llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
              summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
              summary = summary_chain.run(chunks)
              st.success("✅ Summary generated")
              st.write(summary)                   

