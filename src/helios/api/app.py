from contextlib import asynccontextmanager

from fastapi import FastAPI

from helios.api.routes import router
from helios.config import get_config
from helios.runtime.client import SchedulerClient
from helios.runtime.frontend import TextGenerator
from helios.runtime.worker import Tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    app.state.generator = TextGenerator(
        Tokenizer.load(config),
        SchedulerClient(config),
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Helios", lifespan=lifespan)
    app.include_router(router)
    return app
