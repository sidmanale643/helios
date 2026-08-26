from pydantic import BaseModel


class GenerateResponse(BaseModel):
    text: str
