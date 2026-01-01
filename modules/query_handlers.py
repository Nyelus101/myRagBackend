from logger import Logger
from langchain_core.runnables import Runnable # Import to check if it's a Runnable
import os

logger = Logger()


def query_chain(chain, user_input: str):
    try:
        logger.debug(f"Running chain for input: {user_input}")

        # The new chain structure (LCEL) expects the input in a dictionary format 
        # like {"question": ...} and is invoked using .invoke() or .ainvoke().
        
        # Check if the chain is an LCEL Runnable.
        if isinstance(chain, Runnable):
            # Runnables must be executed with .invoke(), and the input format 
            # should match the first component of the chain (which expects "question").
            result = chain.invoke({"question": user_input})
            
        else:
            # Fallback for old style chains (e.g., custom Python functions)
            try:
                result = chain(user_input)
            except TypeError:
                result = chain({"query": user_input})
        
        source_docs = result.get("source_documents", []) if isinstance(result, dict) else []

        #  Deduplicate + strip path + keep only filename
        sources = sorted({
            os.path.basename(doc.metadata.get("source", ""))
            for doc in source_docs
            if doc.metadata.get("source")
        })


        # Normalize result keys (expecting 'result' and 'source_documents')
        # Note: The LCEL chain already returns a dict with 'result' and 'source_documents'
        response = {
            "response": result.get('result') if isinstance(result, dict) else str(result),
            # "sources": [getattr(doc, 'metadata', {}).get('source', '') for doc in result.get('source_documents', [])] if isinstance(result, dict) else []
            "sources": sources
        }

        logger.debug(f"Chain response: {response}")
        return response
    except Exception:
        logger.exception("Error in query_chain:")
        raise


