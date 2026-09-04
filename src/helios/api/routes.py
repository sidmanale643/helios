import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from helios.api.deps import get_generator
from helios.api.types import (
    ChatCompletionBatchItem,
    ChatCompletionBatchRequest,
    ChatCompletionBatchResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionPromptTokensDetails,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionTimings,
    ChatCompletionUsage,
)
from helios.runtime.frontend import TextGenerator

router = APIRouter()
logger = logging.getLogger("uvicorn.error")
GeneratorDependency = Annotated[TextGenerator, Depends(get_generator)]


def _messages(payload: ChatCompletionRequest) -> list[tuple[str, str]]:
    return [
        (
            "system" if message.role == "developer" else message.role,
            message.content,
        )
        for message in payload.messages
    ]


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
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    logger.info(
        "request_received request_id=%s model=%s messages=%d max_new_tokens=%d",
        request_id,
        payload.model,
        len(payload.messages),
        payload.max_tokens,
    )
    if payload.model != generator.model_id:
        logger.warning(
            "request_rejected request_id=%s reason=model_not_loaded requested_model=%s",
            request_id,
            payload.model,
        )
        raise HTTPException(
            status_code=404,
            detail=f"Model '{payload.model}' is not loaded. Use '{generator.model_id}'.",
        )
    try:
        result = await run_in_threadpool(
            generator.run_chat,
            _messages(payload),
            payload.sampling(),
            request_id,
        )
    except ValueError as error:
        logger.warning(
            "request_rejected request_id=%s reason=invalid_request detail=%s",
            request_id,
            error,
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception:
        logger.exception("request_failed request_id=%s", request_id)
        raise
    return ChatCompletionResponse(
        id=request_id,
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
        timings=ChatCompletionTimings(
            tokenize_seconds=result.tokenize_seconds,
            queue_seconds=result.queue_seconds,
            prefix_lookup_seconds=result.prefix_lookup_seconds,
            restore_seconds=result.restore_seconds,
            prefill_seconds=result.prefill_seconds,
            decode_seconds=result.decode_seconds,
            store_seconds=result.store_seconds,
        ),
    )


@router.post("/v1/chat/completions/batch", response_model=ChatCompletionBatchResponse)
async def chat_completions_batch(
    payload: ChatCompletionBatchRequest,
    generator: GeneratorDependency,
) -> ChatCompletionBatchResponse:
    if any(request.model != generator.model_id for request in payload.requests):
        raise HTTPException(
            status_code=404,
            detail=f"Every request must use the loaded model '{generator.model_id}'.",
        )
    try:
        result = await run_in_threadpool(
            generator.run_chat_batch,
            [_messages(request) for request in payload.requests],
            [request.sampling() for request in payload.requests],
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ChatCompletionBatchResponse(
        model=generator.model_id,
        items=[
            ChatCompletionBatchItem(
                index=index,
                content=text,
                finish_reason="stop" if finish_reason == "eos" else "length",
                prompt_tokens=result.prompt_tokens[index],
                completion_tokens=result.completion_tokens[index],
            )
            for index, (text, finish_reason) in enumerate(
                zip(result.texts, result.finish_reasons, strict=True)
            )
        ],
    )
