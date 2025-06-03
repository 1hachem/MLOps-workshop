# ML deployment with huggingface, fastapi, docker and ngrok

## Requirements

- uv: (to manage python libraries and virtual environment)
- docker: (to containerize our application)
- docker: compose (to run our containers together in the same network)
- go-task: (a helper cli tool to organize repetitive commands)

### Optional requirements

- devbox: You can install devbox to avoid having to install these tools manually

## OPTIONAL installing devbox

### To install devbox on ubuntu

```bash
curl -fsSL https://get.jetify.com/devbox | bash
```

### to install devbox on windows with WSL2

You can use Devbox on a Windows machine using Windows Subsystem for Linux 2.

To install WSL2 with the default Ubuntu distribution, open Powershell or Windows Command Prompt as an administrator, and run:

```bash
wsl --install
```

If WSL2 is already installed, you can install Ubuntu by running

```bash
wsl --install -d Ubuntu
```

If you are running an older version of Windows, you may need to follow the manual installation steps to enable virtualization and WSL2 on your system. See the official docs for more details

Run the following script in your WSL2 terminal as a non-root user to install Devbox.

```bash
curl -fsSL https://get.jetify.com/devbox | bash
```

Devbox requires the Nix Package Manager. If Nix is not detected on your machine when running a command, Devbox will automatically install it in single user mode for WSL2. Don't worry: You can use Devbox without needing to learn the Nix Language.

# Repository structure

```
├── src
│   ├── main.py //main script for startin the api and define its endpoints
│   ├── models //module to organize pydantic models used for validation
│   │   ├── generation.py
│   │   └── __init__.py
│   └── utils //module to orgnize utility functions
│       └── utils.py
├── docker
│   └── Dockerfile //Dockerfile for starting our inference api
├── docker-compose.yml //compose file for startin inference api and ngrok agent
├── README.md //this readme
├── Taskfile.yml //task is a helper cli tool to organize useful commands
├─ devbox.json //devbox configuration for development tools
├── devbox.lock
├── pyproject.toml //deps for our python fastapi app
└── uv.lock
```

# Fastapi app

First we start with the imports

```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline

from src.models import HealthCheckResponse
from src.models.generation import TextGenerationRequest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "gpt2"  # You can change this to any Hugging Face model
MODEL_CACHE_DIR = "./model_cache"
```

Fastapi allow us to define a lifespan function that will execute on startup and on shutdown

```python
# Lifespan events

@asynccontextmanager
async def lifespan(app: FastAPI): # Load the ML model
logger.info("Loading model...")
try: # Load tokenizer and model
app.state.tokenizer = AutoTokenizer.from_pretrained(
MODEL_NAME, cache_dir=MODEL_CACHE_DIR
)
app.state.model = AutoModelForCausalLM.from_pretrained(
MODEL_NAME, cache_dir=MODEL_CACHE_DIR
)

        # Create pipeline for easy inference
        app.state.text_generator = pipeline(
            "text-generation", model=app.state.model, tokenizer=app.state.tokenizer
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

    yield  # The application runs here

    # Clean up the ML models and release the resources
    logger.info("Unloading model...")
    del app.state.model
    del app.state.tokenizer
    del app.state.text_generator
    logger.info("Model unloaded successfully")
```

Here we defin the app

```python
app = FastAPI(
title="MLOps workshop",
description="101 to mlops with fastapi, huggingface, docker and ngrok",
contact={"1hachem": "hachem.betrouni@g.enp.edu.dz"},
lifespan=lifespan,
)
```

root GET endpoint that return general info about our inference api

```python
@app.get("/", tags=["info"])
async def root():
"""Root endpoint with basic information"""
    return {
    "message": "Hugging Face Language Model API",
    "model": MODEL_NAME,
    "endpoints": {
    "health_check": "/health",
    "generate_text": "/predict",
    "docs": "/docs",
    },
    }
```

health check endpoint useful to check the status of our api (useful for monitoring and avoiding downtime)

```python
@app.get("/health", response_model=HealthCheckResponse, tags=["monitoring"])
async def health_check():
    """Endpoint to check the health of the service and model status"""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return {"status": "OK", "model_loaded": model_loaded}
```

now our main prediction endpoint

```python
@app.post("/predict", tags=["prediction"])
async def generate_text(request: TextGenerationRequest):
    """Endpoint to generate text based on the given prompt"""
    if not hasattr(app.state, "text_generator"):
    raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Generate text using the pipeline
            generated_texts = app.state.text_generator(
                request.prompt,
                max_length=request.max_length,
                num_return_sequences=request.num_return_sequences,
                temperature=request.temperature,
            )

            # Extract the generated text from the pipeline output
            results = [text["generated_text"] for text in generated_texts]

            return {
                "prompt": request.prompt,
                "generated_texts": results,
                "parameters": {
                    "max_length": request.max_length,
                    "num_return_sequences": request.num_return_sequences,
                    "temperature": request.temperature,
                },
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
```

and our `__main__` logic for starting the api with the specified port

```python
if __name__ == "__main__":
    import uvicorn
    # Using 0.0.0.0 allows the server to accept connections from any IP address
    # on the network, effectively making it accessible externally.
    # The reload=True setting enables automatic reloading of the server when code changes are detected.
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, reload_dirs=".")
```

## Docker

inside `docker/Dockerfile` we define the docker file for our app image, from this image docker will be able to create an isolated container running
our application

```Dockerfile

# Stage 1: Build stage for creating a venv with uv
FROM python:3.11.6-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ADD ./pyproject.toml ./pyproject.toml
ADD ./uv.lock ./uv.lock
ADD ./src ./src

RUN uv sync --frozen

EXPOSE 8000

CMD ["./.venv/bin/python", "./src/main.py"]
```

```yml
services:
  inference:
    build:
      dockerfile: docker/Dockerfile
      context: .
    container_name: inference_api
    ports:
      - "8000:8000"
    develop:
      watch:
        - action: sync
          path: ./src
          target: /app/src

        - action: rebuild
          path: pyproject.toml

  ngrok:
    image: ngrok/ngrok:latest
    command: "http inference:8000 --url=${NGROK_RESERVED_DOMAIN} -v"
    ports:
      - 4040:4040
    environment:
      NGROK_AUTHTOKEN: ${NGROK_AUTHTOKEN}
```
