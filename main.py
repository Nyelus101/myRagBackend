# from fastapi import FastAPI, UploadFile, File, Form, Request
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List
# # from modules.loadvectorstore import load_vectorstore
# from modules.llm import get_llm_chain
# from modules.query_handlers import query_chain
# from logger import Logger
# logger = Logger()
# # from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from modules.loadvectorstore import PERSIST_DIR
# from modules.pdf_handlers import save_uploaded_files
# from modules.loadvectorstore import load_vectorstore_from_paths
# from modules.llm import get_llm_chain, get_general_llm_chain



# MAX_FILE_SIZE_MB = 20
# MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# # Global Initialization (happens once when the server starts)
# GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
# try:
#     GLOBAL_VECTORSTORE = Chroma(
#         persist_directory=PERSIST_DIR,
#         embedding_function=GLOBAL_EMBEDDINGS
#     )
#     logger.info("Chroma vector store loaded successfully at startup.")
# except Exception as e:
#     logger.error(f"Failed to load vector store at startup: {e}")
#     GLOBAL_VECTORSTORE = None # Handle case where it's not ready yet


# app = FastAPI(title="My RagApp3")

# # allow frontend access

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.middleware("http")
# async def catch_exceptions_middleware(request: Request, call_next):
#     try:
#         return await call_next(request)
#     except Exception as exc:
#         logger.exception("Unhandled exception:")
#         return JSONResponse(
#             status_code=500,
#             content={"error": str(exc)},
#         )


# @app.post("/upload_pdfs/")
# async def upload_pdfs(files: List[UploadFile] = File(...)):
#     global GLOBAL_VECTORSTORE

#     try:
#         logger.info(f"Received {len(files)} files")

#         # 1️⃣ Validate file sizes ONLY (no reading twice)
#         for file in files:
#             file_content = await file.read()
#             if len(file_content) > MAX_FILE_SIZE_BYTES:
#                 raise ValueError(
#                     f"File '{file.filename}' exceeds {MAX_FILE_SIZE_MB}MB"
#                 )
#             await file.seek(0)

#         # 2️⃣ Save PDFs once
#         paths = save_uploaded_files(files)

#         # 3️⃣ Load PDFs into Chroma
#         load_vectorstore_from_paths(paths)

#         # 4️⃣ 🔥 RELOAD GLOBAL VECTORSTORE (critical)
#         GLOBAL_VECTORSTORE = Chroma(
#             persist_directory=PERSIST_DIR,
#             embedding_function=GLOBAL_EMBEDDINGS
#         )

#         return {"message": "Files processed and vector store updated."}

#     except ValueError as ve:
#         logger.error(f"Validation Error: {ve}")
#         return JSONResponse(status_code=400, content={"error": str(ve)})

#     except Exception as e:
#         logger.exception("Error during pdf upload")
#         return JSONResponse(status_code=500, content={"error": str(e)})





# # @app.post("/upload_pdfs/")
# # async def upload_pdfs(files: List[UploadFile] = File(...)):
# #     global GLOBAL_VECTORSTORE
# #     try:
# #         logger.info(f"Received {len(files)} files")
        
# #         # New List to hold validated files and their contents
# #         validated_files = [] 
        
# #         for file in files:
# #             file_content = await file.read() # Read content into memory
            
# #             # Reset the file pointer after reading, 
# #             # so it can be re-read later by loadvectorstore if needed, 
# #             # though loadvectorstore will need modification to handle this
# #             await file.seek(0) 

# #             # --- File Size Check ---
# #             if len(file_content) > MAX_FILE_SIZE_BYTES:
# #                 raise ValueError(f"File '{file.filename}' exceeds the size limit of {MAX_FILE_SIZE_MB}MB.")
            
# #             validated_files.append(file) # Add to the list if check passes
        
# #         # Pass the validated files list to the processing function
# #         load_vectorstore(validated_files) 

        
# #         GLOBAL_VECTORSTORE = Chroma(
# #             persist_directory=PERSIST_DIR,
# #             embedding_function=GLOBAL_EMBEDDINGS
# #         )

        
# #         return {"message": "Files processed and vector store updated."}
        
# #     except ValueError as ve:
# #         # Handle specific validation errors
# #         logger.error(f"Validation Error: {ve}")
# #         return JSONResponse(status_code=400, content={"error": str(ve)})
        
# #     except Exception as e:
# #         logger.exception("Error during pdf upload")
# #         return JSONResponse(status_code=500, content={"error": str(e)})


# # @app.post("/ask/")
# # async def ask_question(question: str = Form(...),
# #     mode: str = Form("document")):
# #     if GLOBAL_VECTORSTORE is None:
# #         return JSONResponse(status_code=503, content={"error": "Vector store not initialized."})

# #     try:
# #         # Pass the pre-initialized global vector store
# #         chain = get_llm_chain(GLOBAL_VECTORSTORE) 
# #         result = query_chain(chain, question)
# #         logger.info("query successful")
# #         return result
# #     except Exception as e:
# #         logger.exception("Error processing question")
# #         return JSONResponse(status_code=500, content={"error": str(e)})


# @app.post("/ask/")
# async def ask_question(
#     question: str = Form(...),
#     mode: str = Form("document"),
#     topic: str | None = Form(None)
# ):
#     try:
#         #  Anchor follow-up questions to a product/topic
#         if topic and topic.lower() not in question.lower():
#             question = f"{question} for {topic}"



#         if mode == "general":
#             chain = get_general_llm_chain()
#             result = chain.invoke({"question": question})

#             return {
#                 "response": result,
#                 "sources": []
#             }

#         # Default: document-only RAG
#         if GLOBAL_VECTORSTORE is None:
#             return JSONResponse(
#                 status_code=503,
#                 content={"error": "Vector store not initialized."}
#             )

#         chain = get_llm_chain(GLOBAL_VECTORSTORE)
#         result = query_chain(chain, question)
#         logger.info("query successful")
#         return result

#     except Exception as e:
#         logger.exception("Error processing question")
#         return JSONResponse(status_code=500, content={"error": str(e)})





# @app.get("/test")
# async def test():
#     return {"message": "API is working!"}


# @app.get("/")
# def root():
#     return {"status": "ok"}



















# for pinecone

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.llm import get_llm_chain, get_general_llm_chain
from modules.query_handlers import query_chain
from logger import Logger
from modules.loadvectorstore import load_vectorstore_from_paths, embeddings
from modules.pdf_handlers import save_uploaded_files

logger = Logger()

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Global Initialization (happens once)
GLOBAL_EMBEDDINGS = embeddings
GLOBAL_VECTORSTORE = None  # Will load after first upload

app = FastAPI(title="My RagApp3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled exception:")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global GLOBAL_VECTORSTORE

    try:
        logger.info(f"Received {len(files)} files")

        # Validate file sizes
        for file in files:
            file_content = await file.read()
            if len(file_content) > MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"File '{file.filename}' exceeds {MAX_FILE_SIZE_MB}MB"
                )
            await file.seek(0)

        # Save PDFs
        paths = save_uploaded_files(files)

        # Load PDFs into Pinecone
        GLOBAL_VECTORSTORE = load_vectorstore_from_paths(paths)

        return {"message": "Files processed and vector store updated."}

    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        return JSONResponse(status_code=400, content={"error": str(ve)})

    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask/")
async def ask_question(
    question: str = Form(...),
    mode: str = Form("document"),
    topic: str | None = Form(None)
):
    try:
        if topic and topic.lower() not in question.lower():
            question = f"{question} for {topic}"

        if mode == "general":
            chain = get_general_llm_chain()
            result = chain.invoke({"question": question})
            return {"response": result, "sources": []}

        if GLOBAL_VECTORSTORE is None:
            return JSONResponse(status_code=503, content={"error": "Vector store not initialized."})

        chain = get_llm_chain(GLOBAL_VECTORSTORE)
        result = query_chain(chain, question)
        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/test")
async def test():
    return {"message": "API is working!"}


@app.get("/")
def root():
    return {"status": "ok"}
