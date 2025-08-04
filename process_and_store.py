import os
import sys
import argparse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Setup OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def process_and_store(file_path):
    print(f"🔍 Loading file: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    print(f"📄 Loaded {len(pages)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    
    print(f"🧠 Total chunks: {len(chunks)}. Generating embeddings and storing...")
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")
    db.persist()
    print("✅ Embeddings stored successfully in ./db")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and embed a PDF file")
    parser.add_argument("--file", required=True, help="Path to PDF file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    process_and_store(args.file)
