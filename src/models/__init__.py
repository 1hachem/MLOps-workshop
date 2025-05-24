from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
