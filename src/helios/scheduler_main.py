from helios.config import get_config
from helios.runtime.engine import Engine
from helios.runtime.scheduler import SchedulerServer


def main() -> None:
    config = get_config()
    server = SchedulerServer(Engine(config), config.scheduler_endpoint)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
