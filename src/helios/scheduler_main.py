from helios.config import HeliosConfig, get_config
from helios.runtime.engine import Engine
from helios.runtime.scheduler import SchedulerServer


def main(config: HeliosConfig | None = None) -> None:
    config = config or get_config()
    server = SchedulerServer(Engine(config), config.scheduler_endpoint)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
