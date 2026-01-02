



import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

# -------------------- LLM --------------------
# 1. Global Initialization of LLM (Happens only once on startup)
# This avoids re-instantiating the LLM object on every request.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

LLM = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# -------------------- PROMPT --------------------
# 2. Define the RAG Prompt Template (Used in all queries)
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert assistant. Use *only* the following context to answer the user's question. 
**Do not use any phrases like 'Based on the provided context,' 'According to the documents,' or similar qualifiers in your answer.**
# If the context does not contain the answer, simply state, "The required information is not available in the documents."
If you cannot find the answer in the context, state that you don't know, do not try to make up an answer.

Context:
{context}

Question: {question}

Answer:
""")


# -------------------- GENERAL PROMPT --------------------
GENERAL_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful and knowledgeable assistant.
Answer the user's question clearly and accurately.

Question: {question}

Answer:
""")

def get_general_llm_chain():
    return (
        RunnablePassthrough()
        | GENERAL_PROMPT
        | LLM
        | StrOutputParser()
    )




# -------------------- CHAIN --------------------
# 3. Refactored function to return a scalable LCEL chain
def get_llm_chain(vectorstore):
    """
    Creates and returns a LangChain Runnable for the RAG query process.
    The LLM is already initialized globally.

    Hybrid RAG:
    - BM25 (keyword)
    - Chroma (semantic)
    - Ensemble retriever
    """
    
    # The retriever is still initialized inside the function 
    # as it depends on the vectorstore passed from main.py.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Function to format the retrieved documents into a single string for the prompt.
    # def format_docs(docs):
    #     # We join the page_content of the retrieved documents with a double newline
    #     return "\n\n".join(doc.page_content for doc in docs)

    def format_docs(docs):
        clean_chunks = []
        for doc in docs:
            text = doc.page_content.replace("\n\n", " ").replace("\n-", "\n•")
            clean_chunks.append(text)
        return " ".join(clean_chunks)

    # --- LangChain Expression Language (LCEL) Chain ---
    # This defines the execution pipeline clearly and efficiently.
    # 
    rag_chain = (
        # 1. Get the question from the input
        # 2. Retrieve documents (retriever)
        # 3. Format documents (format_docs)
        RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
        # 4. Prepare the final prompt (RAG_PROMPT) using both context and question
        | RAG_PROMPT
        # 5. Invoke the globally initialized LLM
        | LLM
        # 6. Parse the LLM's output into a simple string
        | StrOutputParser()
    )

    # The chain needs to return both the answer and the source documents.
    # We define a structured output using RunnableParallel to execute both retrieval and generation.
    final_rag_chain = RunnablePassthrough.assign(
        # The 'source_documents' are retrieved first
        source_documents=itemgetter("question") | retriever
    ) | {
        # 'result' uses the output of 'source_documents' and the 'question' to run the generation chain
        "result": RunnablePassthrough.assign(
            context=itemgetter("source_documents") | RunnableLambda(format_docs)
        )
        | RAG_PROMPT
        | LLM
        | StrOutputParser(),
        # 'source_documents' is passed through directly from the previous step
        "source_documents": itemgetter("source_documents"),
    }
    
    # We return the structured chain which will output a dictionary with 'result' and 'source_documents'
    return final_rag_chain






















# import os
# from dotenv import load_dotenv
# from operator import itemgetter

# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# # from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
# # from langchain.retrievers import BM25Retriever, EnsembleRetriever
# from langchain_community.retrievers import BM25Retriever
# # from langchain_community.retrievers import EnsembleRetriever, BM25Retriever
# from langchain import EnsembleRetriever
# from langchain_core.documents import Document

# load_dotenv()

# # -------------------- LLM --------------------
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# LLM = ChatGroq(
#     groq_api_key=GROQ_API_KEY,
#     model_name="llama-3.1-8b-instant"
# )

# # -------------------- PROMPT --------------------
# RAG_PROMPT = ChatPromptTemplate.from_template("""
# You are an expert assistant. Use *only* the following context to answer the user's question.
# If the context does not contain the answer, simply state:
# "The required information is not available in the documents."

# Context:
# {context}

# Question: {question}

# Answer:
# """)

# # -------------------- CHAIN --------------------
# def get_llm_chain(vectorstore):
#     """
#     Hybrid RAG:
#     - BM25 (keyword)
#     - Chroma (semantic)
#     - Ensemble retriever
#     """

#     # 1️⃣ Semantic retriever (Chroma)
#     semantic_retriever = vectorstore.as_retriever(
#         search_kwargs={"k": 6}
#     )

#     # 2️⃣ Keyword retriever (BM25)
#     # We pull ALL docs currently in Chroma
#     docs = vectorstore.get()["documents"]
#     metadatas = vectorstore.get()["metadatas"]

#     from langchain_core.documents import Document
#     # from langchain.schema import- Document
#     bm25_docs = [
#         Document(page_content=doc, metadata=meta)
#         for doc, meta in zip(docs, metadatas)
#     ]

#     bm25_retriever = BM25Retriever.from_documents(bm25_docs)
#     bm25_retriever.k = 6

#     # 3️⃣ Hybrid ensemble
#     hybrid_retriever = EnsembleRetriever(
#         retrievers=[bm25_retriever, semantic_retriever],
#         weights=[0.5, 0.5]  # balanced (can tune later)
#     )

#     # -------------------- Helpers --------------------
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     # -------------------- FINAL LCEL --------------------
#     final_rag_chain = RunnablePassthrough.assign(
#         source_documents=itemgetter("question") | hybrid_retriever
#     ) | {
#         "result": RunnablePassthrough.assign(
#             context=itemgetter("source_documents") | RunnableLambda(format_docs)
#         )
#         | RAG_PROMPT
#         | LLM
#         | StrOutputParser(),
#         "source_documents": itemgetter("source_documents"),
#     }

#     return final_rag_chain
