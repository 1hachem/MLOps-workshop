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


# Lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Loading model...")
    try:
        # Load tokenizer and model
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


app = FastAPI(
    title="MLOps workshop",
    description="101 to mlops with fastapi, huggingface, docker and ngrok",
    contact={"1hachem": "hachem.betrouni@g.enp.edu.dz"},
    lifespan=lifespan,
)


# Root endpoint
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


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["monitoring"])
async def health_check():
    """Endpoint to check the health of the service and model status"""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return {"status": "OK", "model_loaded": model_loaded}


# Prediction endpoint
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, reload_dirs=".")
