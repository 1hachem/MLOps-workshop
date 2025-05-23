from fastapi import FastAPI

from src.utils.utils import print_hello

app = FastAPI(
    title="MLOps workshop",
    description="101 to mlops with fastapi, huggingface, docker and ngrok",
)


@app.get("/")
async def root():
    print_hello()
    return {"message": "Hello from mlops-workshop!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
