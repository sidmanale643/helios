from pydantic import BaseModel, Field


class Sampling(BaseModel):
    max_new_tokens: int = Field(default=256, ge=1, le=2_048)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
