from helios.config import get_config
from helios.runtime.client import SchedulerClient
from helios.runtime.frontend import TextGenerator
from helios.runtime.types import GenerateRequest
from helios.runtime.worker import Tokenizer

prompts = [
    "Explain solar power simply.",
    "Explain photosynthesis simply.",
]


def main() -> None:
    config = get_config()
    generator = TextGenerator(
        Tokenizer.load(config),
        SchedulerClient(config),
    )
    responses = [generator.run(GenerateRequest(text=prompt)) for prompt in prompts]

    for prompt, response in zip(prompts, responses, strict=True):
        print(f"Prompt: {prompt}\n\nResponse: {response}\n")


if __name__ == "__main__":
    main()
