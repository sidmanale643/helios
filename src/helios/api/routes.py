import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from helios.api.deps import get_generator
from helios.api.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionPromptTokensDetails,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)
from helios.runtime.frontend import TextGenerator

router = APIRouter()
GeneratorDependency = Annotated[TextGenerator, Depends(get_generator)]


@router.get("/health")
async def health(
    generator: GeneratorDependency,
) -> dict[str, object]:
    return await run_in_threadpool(generator.health)


@router.get("/internal/cache")
async def prefix_cache_state(generator: GeneratorDependency) -> dict[str, object]:
    return await run_in_threadpool(generator.engine.prefix_cache_snapshot)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    payload: ChatCompletionRequest,
    generator: GeneratorDependency,
) -> ChatCompletionResponse:
    if payload.model != generator.model_id:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{payload.model}' is not loaded. Use '{generator.model_id}'.",
        )
    try:
        result = await run_in_threadpool(
            generator.run_chat,
            [
                (
                    "system" if message.role == "developer" else message.role,
                    message.content,
                )
                for message in payload.messages
            ],
            payload.sampling(),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=generator.model_id,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(content=result.text),
                finish_reason="stop" if result.finish_reason == "eos" else "length",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
            prompt_tokens_details=ChatCompletionPromptTokensDetails(
                cached_tokens=result.cached_tokens,
            ),
        ),
    )
