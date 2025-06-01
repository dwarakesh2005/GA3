from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI(title="InfoCore Semantic Search API")

# CORS Configuration - This is the key fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Include OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# IITM AI Proxy configuration
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDE5MTVAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.acO3-kXAgc-Q7TWfcThE2JLAsU81PDdvS6iIBfu7ELo"

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: List[str]

def get_embedding(text: str) -> List[float]:
    try:
        url = f"{AIPROXY_BASE_URL}/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        data = {
            "model": "text-embedding-3-small",
            "input": text
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "InfoCore Semantic Search API", "status": "active"}

@app.post("/similarity", response_model=SimilarityResponse)
async def similarity_search(request: SimilarityRequest):
    try:
        if not request.docs:
            raise HTTPException(status_code=400, detail="No documents provided")
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate embeddings for all documents
        doc_embeddings = []
        for doc in request.docs:
            embedding = get_embedding(doc)
            doc_embeddings.append(embedding)
        
        # Generate embedding for the search query
        query_embedding = get_embedding(request.query)
        
        # Convert to numpy arrays for cosine similarity computation
        doc_embeddings_array = np.array(doc_embeddings)
        query_embedding_array = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding_array, doc_embeddings_array)[0]
        
        # Get indices of top 3 most similar documents
        num_results = min(3, len(request.docs))
        top_indices = np.argsort(similarities)[::-1][:num_results]
        
        # Return the actual document contents of top matches
        matches = [request.docs[i] for i in top_indices]
        
        return SimilarityResponse(matches=matches)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Add exception handler for CORS on errors
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
