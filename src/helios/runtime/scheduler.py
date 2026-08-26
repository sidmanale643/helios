from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event

import zmq
from pydantic import ValidationError

from helios.runtime.engine import Engine, ModelMismatchError
from helios.runtime.protocol import (
    ErrorResult,
    GenerateCommand,
    HealthCommand,
    HealthResult,
    Result,
    command_adapter,
)


@dataclass(frozen=True)
class ScheduledCommand:
    route: bytes
    command: GenerateCommand


class Scheduler:

    def __init__(self) -> None:
        self._waiting: deque[ScheduledCommand] = deque()

    def submit(self, route: bytes, command: GenerateCommand) -> None:
        self._waiting.append(ScheduledCommand(route, command))

    def next(self) -> ScheduledCommand:
        return self._waiting.popleft()

    def __bool__(self) -> bool:
        return bool(self._waiting)


class SchedulerServer:
    def __init__(
        self,
        engine: Engine,
        endpoint: str,
        *,
        context: zmq.Context | None = None,
    ) -> None:
        self.engine = engine
        self.endpoint = endpoint
        self.context = context or zmq.Context.instance()
        self.scheduler = Scheduler()

    def serve(self, stop_event: Event | None = None) -> None:
        stop_event = stop_event or Event()
        socket = self.context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        # A single model worker preserves the FIFO order of accepted requests.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="helios-model")
        active: tuple[ScheduledCommand, Future[Result]] | None = None
        try:
            while not stop_event.is_set():
                if dict(poller.poll(10)).get(socket, 0) & zmq.POLLIN:
                    self._receive(socket)
                    self._drain(socket)

                if active is not None and active[1].done():
                    scheduled, future = active
                    self._send(socket, scheduled.route, future.result())
                    active = None

                if active is None and self.scheduler:
                    scheduled = self.scheduler.next()
                    active = (
                        scheduled,
                        executor.submit(self._generate, scheduled.command),
                    )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            poller.unregister(socket)
            socket.close()

    def _receive(self, socket: zmq.Socket) -> None:
        frames = socket.recv_multipart()
        route = frames[0]
        if len(frames) != 2:
            result = ErrorResult(
                request_id="unknown",
                code="invalid_request",
                detail="Scheduler requests must contain exactly one payload frame.",
            )
            self._send(socket, route, result)
            return
        payload = frames[1]
        try:
            command = command_adapter.validate_json(payload)
        except ValidationError as error:
            result = ErrorResult(
                request_id="unknown",
                code="invalid_request",
                detail=str(error),
            )
            self._send(socket, route, result)
            return
        if isinstance(command, HealthCommand):
            self._send(
                socket,
                route,
                HealthResult(
                    request_id=command.request_id,
                    model_id=self.engine.model_id,
                    model_revision=self.engine.model_revision,
                    memory=self.engine.report.as_dict(),
                ),
            )
            return
        self.scheduler.submit(route, command)

    def _drain(self, socket: zmq.Socket) -> None:
        while socket.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            self._receive(socket)

    def _generate(self, command: GenerateCommand) -> Result:
        try:
            return self.engine.run(command)
        except ModelMismatchError as error:
            return ErrorResult(
                request_id=command.request_id,
                code="model_mismatch",
                detail=str(error),
            )
        except ValueError as error:
            return ErrorResult(
                request_id=command.request_id,
                code="invalid_request",
                detail=str(error),
            )
        except (OSError, RuntimeError):
            return ErrorResult(
                request_id=command.request_id,
                code="internal_error",
                detail="The model executor failed while processing the request.",
            )

    @staticmethod
    def _send(socket: zmq.Socket, route: bytes, result: Result) -> None:
        socket.send_multipart([route, result.model_dump_json().encode("utf-8")])
