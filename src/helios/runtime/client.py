from uuid import uuid4

import zmq
from pydantic import ValidationError

from helios.config import HeliosConfig
from helios.runtime.protocol import (
    Command,
    ErrorResult,
    GenerateCommand,
    GenerateResult,
    HealthCommand,
    HealthResult,
    Result,
    result_adapter,
)
from helios.runtime.types import Sampling


class SchedulerClient:
    def __init__(
        self,
        config: HeliosConfig,
        *,
        context: zmq.Context | None = None,
    ) -> None:
        self.endpoint = config.scheduler_endpoint
        self.timeout_ms = config.scheduler_timeout_ms
        self.context = context or zmq.Context.instance()

    def generate(
        self,
        *,
        model_id: str,
        model_revision: str,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
    ) -> GenerateResult:
        command = GenerateCommand(
            request_id=uuid4().hex,
            model_id=model_id,
            model_revision=model_revision,
            input_ids=input_ids,
            eos_token_id=eos_token_id,
            sampling=sampling,
        )
        result = self._request(command, self.timeout_ms)
        if not isinstance(result, GenerateResult):
            raise SchedulerProtocolError("Scheduler returned the wrong response type.")
        return result

    def health(self, timeout_ms: int = 1_000) -> HealthResult:
        command = HealthCommand(request_id=uuid4().hex)
        result = self._request(command, min(timeout_ms, self.timeout_ms))
        if not isinstance(result, HealthResult):
            raise SchedulerProtocolError("Scheduler returned the wrong response type.")
        return result

    def _request(self, command: Command, timeout_ms: int) -> Result:
        socket = self.context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        try:
            socket.send(command.model_dump_json().encode("utf-8"))
            if not socket.poll(timeout_ms, zmq.POLLIN):
                raise SchedulerUnavailable(
                    f"Scheduler did not respond within {timeout_ms} ms."
                )
            payload = socket.recv()
        except zmq.ZMQError as error:
            raise SchedulerUnavailable(
                f"Scheduler transport failed: {error}"
            ) from error
        finally:
            socket.close()

        try:
            result = result_adapter.validate_json(payload)
        except ValidationError as error:
            raise SchedulerProtocolError(
                "Scheduler returned an invalid response."
            ) from error
        if result.request_id != command.request_id:
            raise SchedulerProtocolError(
                "Scheduler response request_id does not match."
            )
        if isinstance(result, ErrorResult):
            raise SchedulerRemoteError(result.code, result.detail)
        return result


class SchedulerUnavailable(RuntimeError):
    pass


class SchedulerProtocolError(RuntimeError):
    pass


class SchedulerRemoteError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
