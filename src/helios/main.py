import signal
from multiprocessing import Process, get_context

import uvicorn

from helios.api.app import create_app

app = create_app()


def main() -> None:
    server = get_context("spawn").Process(
        target=_serve,
        name="helios-server",
        daemon=True,
    )
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt)
    try:
        server.start()
        while server.is_alive():
            server.join(timeout=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _stop(server)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    if server.exitcode not in (0, -signal.SIGTERM):
        raise SystemExit(server.exitcode)


def _serve() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def _stop(process: Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join()


if __name__ == "__main__":
    main()
