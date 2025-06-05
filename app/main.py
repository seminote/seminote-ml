from fastapi import FastAPI

app = FastAPI(title="Seminote ML Services", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "seminote-ml"}

@app.get("/")
async def root():
    return {"service": "Seminote ML Services", "docs": "/docs"}
