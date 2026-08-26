from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class Sampling(BaseModel):
    max_new_tokens: int = Field(default=256, ge=1, le=2_048)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(
        validation_alias=AliasChoices("text", "prompt"),
        min_length=1,
        max_length=20_000,
    )
    sampling: Sampling = Field(default_factory=Sampling)

    @model_validator(mode="before")
    @classmethod
    def sampling_from_flat(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = value.copy()
        sampling = dict(value.get("sampling", {}))
        for name in Sampling.model_fields:
            if name in value:
                sampling[name] = value.pop(name)
        value["sampling"] = sampling
        return value


class GenerateBatch(BaseModel):
    requests: list[GenerateRequest] = Field(min_length=1, max_length=32)
