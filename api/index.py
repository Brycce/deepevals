from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/api/health")
async def health():
    return {"healthy": True}

# Wrap FastAPI with Mangum for serverless
handler = Mangum(app, lifespan="off")
