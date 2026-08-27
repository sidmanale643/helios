from multiprocessing import Process, get_context
from time import monotonic, sleep

import uvicorn

from helios.api.app import create_app
from helios.config import HeliosConfig, get_config
from helios.runtime.client import SchedulerClient, SchedulerUnavailable
from helios.scheduler_main import main as scheduler_main

app = create_app()


def main() -> None:
    config = get_config()
    print(f"\nHelios\n  scheduler  loading {config.model_id}\n", flush=True)
    scheduler = get_context("spawn").Process(
        target=_run_scheduler,
        args=(config,),
        name="helios-scheduler",
    )
    scheduler.start()
    try:
        _wait_for_scheduler(config, scheduler)
        print(
            "  scheduler  ready\n"
            "  api        starting at http://127.0.0.1:8000\n"
            "  docs       http://127.0.0.1:8000/docs\n",
            flush=True,
        )
        uvicorn.run(app, host="127.0.0.1", port=8000)
    finally:
        if scheduler.is_alive():
            scheduler.terminate()
        scheduler.join()


def _run_scheduler(config: HeliosConfig) -> None:
    scheduler_main(config)


def _wait_for_scheduler(config: HeliosConfig, scheduler: Process) -> None:
    client = SchedulerClient(config)
    deadline = monotonic() + config.scheduler_timeout_ms / 1_000
    while scheduler.is_alive() and monotonic() < deadline:
        try:
            client.health(timeout_ms=250)
            return
        except SchedulerUnavailable:
            sleep(0.1)
    if scheduler.exitcode is not None:
        raise RuntimeError(f"Scheduler exited during startup with code {scheduler.exitcode}.")
    raise RuntimeError(
        f"Scheduler did not become ready within {config.scheduler_timeout_ms} ms."
    )


if __name__ == "__main__":
    main()
