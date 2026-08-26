import uvicorn

from helios.api.app import create_app

app = create_app()


def main() -> None:
    uvicorn.run("helios.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
