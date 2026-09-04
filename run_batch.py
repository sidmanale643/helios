from helios.config import get_config
from helios.runtime.engine import Engine
from helios.runtime.frontend import TextGenerator
from helios.runtime.types import GenerateBatch, GenerateRequest, Sampling
from helios.runtime.worker import Tokenizer

prompts = [
    "Explain solar power simply.",
    "Explain photosynthesis simply.",
]


def main() -> None:
    config = get_config()
    generator = TextGenerator(
        Tokenizer.load(config),
        Engine(config),
    )
    sampling = Sampling()
    responses = generator.run_batch(
        GenerateBatch(
            requests=[
                GenerateRequest(text=prompt, sampling=sampling) for prompt in prompts
            ]
        )
    )

    for prompt, response in zip(prompts, responses, strict=True):
        print(f"Prompt: {prompt}\n\nResponse: {response}\n")


if __name__ == "__main__":
    main()
