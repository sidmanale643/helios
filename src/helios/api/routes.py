from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from helios.api.deps import get_generator
from helios.api.types import GenerateResponse
from helios.runtime.client import (
    SchedulerProtocolError,
    SchedulerRemoteError,
    SchedulerUnavailable,
)
from helios.runtime.frontend import TextGenerator
from helios.runtime.types import GenerateRequest

router = APIRouter()
GeneratorDependency = Annotated[TextGenerator, Depends(get_generator)]


@router.get("/health")
async def health(
    generator: GeneratorDependency,
) -> dict[str, object]:
    try:
        result = await run_in_threadpool(generator.health)
    except (SchedulerUnavailable, SchedulerProtocolError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "status": result.status,
        "model": result.model_id,
        "model_revision": result.model_revision,
        "memory": result.memory,
    }


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    payload: GenerateRequest,
    generator: GeneratorDependency,
) -> GenerateResponse:
    try:
        text = await run_in_threadpool(generator.run, payload)
    except SchedulerRemoteError as error:
        status_code = 422 if error.code == "invalid_request" else 503
        raise HTTPException(status_code=status_code, detail=error.detail) from error
    except (SchedulerUnavailable, SchedulerProtocolError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return GenerateResponse(text=text)
