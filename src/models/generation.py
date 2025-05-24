from typing import Optional

from pydantic import BaseModel


# Request models
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    num_return_sequences: Optional[int] = 1
    temperature: Optional[float] = 1.0
