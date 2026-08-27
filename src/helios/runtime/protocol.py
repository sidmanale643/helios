from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from helios.runtime.types import Sampling

PROTOCOL_VERSION = 1
TokenId = Annotated[int, Field(strict=True, ge=0)]


class GenerateCommand(BaseModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    operation: Literal["generate"] = "generate"
    request_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    input_ids: list[TokenId] = Field(min_length=1, max_length=40_960)
    eos_token_id: TokenId
    sampling: Sampling = Field(default_factory=Sampling)


class HealthCommand(BaseModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    operation: Literal["health"] = "health"
    request_id: str = Field(min_length=1, max_length=128)


Command = Annotated[GenerateCommand | HealthCommand, Field(discriminator="operation")]
command_adapter = TypeAdapter(Command)


class GenerationTiming(BaseModel):
    prefill_seconds: float = Field(ge=0)
    inter_token_seconds: list[float] = Field(default_factory=list)


class GenerateResult(BaseModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    operation: Literal["generate"] = "generate"
    request_id: str
    output_ids: list[TokenId]
    finish_reason: Literal["eos", "length"]
    timing: GenerationTiming | None = None


class HealthResult(BaseModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    operation: Literal["health"] = "health"
    request_id: str
    status: Literal["ok"] = "ok"
    model_id: str
    model_revision: str
    memory: dict[str, object]


class ErrorResult(BaseModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    operation: Literal["error"] = "error"
    request_id: str
    code: Literal["invalid_request", "model_mismatch", "internal_error"]
    detail: str


Result = Annotated[
    GenerateResult | HealthResult | ErrorResult,
    Field(discriminator="operation"),
]
result_adapter = TypeAdapter(Result)
