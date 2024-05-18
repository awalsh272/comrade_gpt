import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.embeddings import SentenceTransformerEmbeddings

pdf_path = "src\\data"
chroma_dir = "src\\chroma_db"


def get_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


def load_chunk_persist_pdf(
    pdf_folder_path: str = pdf_path, chroma_dir: str = chroma_dir
) -> Chroma:
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("chroma_db")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=get_embeddings(),
        persist_directory=chroma_dir,
    )
    return vectordb


def get_existing_vectordb(vector_db_path: str = chroma_dir):
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    print("got vector store")
    return vectorstore