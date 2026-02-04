from fastapi import FastAPI, File, HTTPException, UploadFile
import uvicorn
from pathlib import Path
# import shutil

from multimodal_search_engine.embedder import CLIPEmbedder
from multimodal_search_engine.qdrant_search_client import QdrantSearchClient

app = FastAPI()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

@app.post("/search")
async def search():
    '''
    поиск изображения по текстовому запросу
    '''
    return 0



if __name__ == "__main__":
    uvicorn.run(app=app, reload=True, port=8000)
