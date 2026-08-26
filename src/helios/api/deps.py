from fastapi import HTTPException, Request

from helios.runtime.frontend import TextGenerator


def get_generator(request: Request) -> TextGenerator:
    generator: TextGenerator | None = getattr(request.app.state, "generator", None)
    if generator is None:
        raise HTTPException(status_code=503, detail="Tokenizer frontend is not ready")
    return generator
