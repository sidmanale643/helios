import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from helios.api.routes import router
from helios.config import get_config
from helios.runtime.engine import Engine
from helios.runtime.frontend import TextGenerator
from helios.runtime.worker import Tokenizer

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    logger.info(
        "startup_model_loading model=%s torch_compile=%s",
        config.model_id,
        config.torch_compile,
    )
    tokenizer = Tokenizer.load(config)
    logger.info(
        "startup_tokenizer_loaded model=%s revision=%s",
        tokenizer.model_id,
        tokenizer.model_revision,
    )
    engine = Engine(config)
    logger.info(
        "startup_model_loaded model=%s revision=%s",
        engine.model_id,
        engine.model_revision,
    )
    generator = TextGenerator(tokenizer, engine)
    if config.torch_compile:
        logger.info("startup_compile_warmup_started")
    generator.warm_up_compile()
    if config.torch_compile:
        logger.info("startup_compile_warmup_completed")
    app.state.generator = generator
    logger.info(
        "startup_ready model=%s revision=%s",
        generator.model_id,
        generator.engine.model_revision,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Helios", lifespan=lifespan)
    app.include_router(router)
    return app
