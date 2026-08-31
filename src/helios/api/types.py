from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

from helios.runtime.types import Sampling


class ChatMessage(BaseModel):
    role: Literal["developer", "system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(
        default=256,
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
        ge=1,
        le=2_048,
    )
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    stream: Literal[False] = False

    def sampling(self) -> Sampling:
        return Sampling(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
