import logging
import os
from typing import TYPE_CHECKING

import torch

from helios.config import HeliosConfig

if TYPE_CHECKING:
    from helios.runtime.qwen3.model import Qwen3Model

logger = logging.getLogger("uvicorn.error")


def configure_compilation(model: "Qwen3Model", config: HeliosConfig) -> None:
    if not config.torch_compile:
        return
    if config.compile_diagnostics:
        if "TORCH_LOGS" in os.environ:
            logger.info("torch_compile_diagnostics source=TORCH_LOGS")
        else:
            torch._logging.set_logs(
                dynamo=logging.INFO, graph_breaks=True, recompiles=True,
            )
    model.compile(
        backend="inductor", fullgraph=config.compile_fullgraph, dynamic=True
    )
    model.decode_forward = torch.compile(
        model.decode_forward,
        backend="inductor",
        fullgraph=config.compile_fullgraph,
        dynamic=True,
    )
    logger.info(
        "torch_compile_enabled backend=inductor mode=default dynamic=True fullgraph=%s "
        "diagnostics=%s compilation=lazy",
        config.compile_fullgraph,
        config.compile_diagnostics,
    )
