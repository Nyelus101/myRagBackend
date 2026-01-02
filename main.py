from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.query_handlers import query_chain
from logger import Logger
import shutil
import os
import time
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from modules.pdf_handlers import save_uploaded_files, UPLOAD_DIR
from modules.loadvectorstore import load_vectorstore_from_paths, PERSIST_DIR
from modules.llm import get_llm_chain, get_general_llm_chain


logger = Logger()

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# Global Initialization (happens once when the server starts)
GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
try:
    GLOBAL_VECTORSTORE = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=GLOBAL_EMBEDDINGS
    )
    logger.info("Chroma vector store loaded successfully at startup.")
except Exception as e:
    logger.error(f"Failed to load vector store at startup: {e}")
    GLOBAL_VECTORSTORE = None # Handle case where it's not ready yet


app = FastAPI(title="My RagApp3")

# allow frontend access

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
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global GLOBAL_VECTORSTORE

    try:
        logger.info(f"Received {len(files)} files")

        # 1️ Validate file sizes ONLY (no reading twice)
        for file in files:
            file_content = await file.read()
            if len(file_content) > MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"File '{file.filename}' exceeds {MAX_FILE_SIZE_MB}MB"
                )
            await file.seek(0)

        # 2️ Save PDFs once
        paths = save_uploaded_files(files)

        # 3️ Load PDFs into Chroma
        load_vectorstore_from_paths(paths)

        # 4️ RELOAD GLOBAL VECTORSTORE (critical)
        GLOBAL_VECTORSTORE = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=GLOBAL_EMBEDDINGS
        )

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
        #  Anchor follow-up questions to a product/topic
        if topic and topic.lower() not in question.lower():
            question = f"{question} for {topic}"



        if mode == "general":
            chain = get_general_llm_chain()
            result = chain.invoke({"question": question})

            return {
                "response": result,
                "sources": []
            }

        # Default: document-only RAG
        if GLOBAL_VECTORSTORE is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Vector store not initialized."}
            )

        chain = get_llm_chain(GLOBAL_VECTORSTORE)
        result = query_chain(chain, question)
        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})



# --- Clear storage endpoint ---
@app.post("/clear_storage/")
def clear_storage():
    global GLOBAL_VECTORSTORE
    try:
        # 1️ Close Chroma safely
        if GLOBAL_VECTORSTORE is not None:
            try:
                # Persist any in-memory changes (optional)
                GLOBAL_VECTORSTORE.persist()
                # Explicitly clear internal collection and client references
                GLOBAL_VECTORSTORE._collection = None
                GLOBAL_VECTORSTORE._client = None
            except Exception:
                pass
            # Remove reference so Python GC can clear it
            GLOBAL_VECTORSTORE = None

        # 2️ Wait a moment to ensure all file handles are released
        time.sleep(0.1)

        # 3️ Delete directories
        dirs_to_clear = ["./uploaded_pdfs", "./chroma_store"]
        for d in dirs_to_clear:
            if os.path.exists(d):
                for attempt in range(5):
                    try:
                        shutil.rmtree(d)
                        logger.info(f"Deleted directory: {d}")
                        break
                    except PermissionError:
                        logger.warning(f"PermissionError, retrying... ({attempt+1}/5)")
                        time.sleep(0.2)
            os.makedirs(d, exist_ok=True)
            logger.info(f"Re-created directory: {d}")

        return {"message": "Storage cleared successfully."}

    except Exception as e:
        logger.exception("Failed to clear storage")
        return {"error": str(e)}



# --- Stats Endpoint ---
# ------------------- Stats Endpoint -------------------
@app.get("/stats/")
def get_stats():
    try:
        # Recent uploads
        if os.path.exists(UPLOAD_DIR):
            recent_uploads = [
                f for f in os.listdir(UPLOAD_DIR)
                if f.lower().endswith(".pdf")
            ]
        else:
            recent_uploads = []

        # Vectorstore stats
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
            vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )
            docs_count = vectorstore._collection.count() if vectorstore._collection else 0
            total_vectors = docs_count
        else:
            docs_count = 0
            total_vectors = 0

        # Last updated
        if recent_uploads:
            last_updated_ts = max(
                os.path.getmtime(os.path.join(UPLOAD_DIR, f)) for f in recent_uploads
            )
            last_updated = datetime.fromtimestamp(last_updated_ts).strftime("%Y-%m-%d %H:%M:%S")
        else:
            last_updated = "N/A"

        return {
            "documentsIndexed": docs_count,
            "totalVectors": total_vectors,
            "lastUpdated": last_updated,
            "recentUploads": recent_uploads,
        }
    except Exception as e:
        logger.exception("Failed to get stats")
        return {"error": str(e)}







@app.get("/test")
async def test():
    return {"message": "API is working!"}


@app.get("/")
def root():
    return {"status": "ok"}
