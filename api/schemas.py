from pydantic import BaseModel


class PredictRequest(BaseModel):
    values: list[float]


class PredictResponse(BaseModel):
    predicted_class: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


class GradCAMRequest(BaseModel):
    values: list[float]


class GradCAMResponse(BaseModel):
    heatmap: list[float]
    image_base64: str


class ReferenceResponse(BaseModel):
    waveforms: dict[str, list[float]]
