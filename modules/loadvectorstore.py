# # Saves PDFs to disk
# # Loads PDF content
# # Splits text into chunks
# # Generates embeddings
# # Stores them in Chroma


# import os
# from pathlib import Path
# # from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter




# PERSIST_DIR="./chroma_store"
# UPLOAD_DIR="./uploaded_pdfs"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # def load_vectorstore(uploaded_files):
# #     file_paths = []
# #     for file in uploaded_files:
# #         save_path = Path(UPLOAD_DIR) / file.filename
# #         with open(save_path, "wb") as f:
# #             f.write(file.file.read())
# #         file_paths.append(str(save_path))

# #     docs = []
# #     for path in file_paths:
# #         loader = PyPDFLoader(path)
# #         docs.extend(loader.load())

# #     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
# #     texts = splitter.split_documents(docs)

# #     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

# #     if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
# #         vectorstore = Chroma(
# #             persist_directory=PERSIST_DIR,
# #             embedding_function=embeddings
# #         )
# #         vectorstore.add_documents(texts)
# #         # vectorstore.persist()
# #     else:
# #         vectorstore = Chroma.from_documents(
# #             documents=texts,
# #             embedding=embeddings,
# #             persist_directory=PERSIST_DIR
# #         )
#         # vectorstore.persist()


# def load_vectorstore_from_paths(file_paths: list[str]):
#     docs = []
#     for path in file_paths:
#         loader = PyPDFLoader(path)
#         docs.extend(loader.load())

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=200
#     )
#     texts = splitter.split_documents(docs)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

#     if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
#         vectorstore = Chroma(
#             persist_directory=PERSIST_DIR,
#             embedding_function=embeddings
#         )
#         vectorstore.add_documents(texts)
#     else:
#         Chroma.from_documents(
#             documents=texts,
#             embedding=embeddings,
#             persist_directory=PERSIST_DIR
#         )



#     return vectorstore











# for pinecone

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone  # official Pinecone SDK

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-east1-gcp"
INDEX_NAME = "rag-index"

UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone once
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create index if it doesn't exist
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=384)  # MiniLM-L12-v2 embedding dim

# Initialize embeddings globally
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")


def load_vectorstore_from_paths(file_paths: list[str]):
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    texts = splitter.split_documents(docs)

    # Create or load Pinecone vectorstore
    vectorstore = Pinecone.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    return vectorstore
