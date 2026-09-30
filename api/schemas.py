# this file defines the structure of the json response returned by FastAPI endpoint

from pydantic import BaseModel

# health response

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# prediction response

class predicitonresponse(BaseModel):
    prediction: int
    diagnosis: str
    stiction_probability: float
    inference_time_seconds: float

class modelinforesponse(BaseModel):
    model_name: str
    image_type: str
    encoder: str
    classifier: str
    image_size: str


