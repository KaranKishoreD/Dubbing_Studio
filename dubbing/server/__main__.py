"""
Start the web studio:

    cd dubbing
    python -m server                  # http://127.0.0.1:8000
    python -m server --port 9000
    python -m server --host 0.0.0.0   # reachable from other machines

To share it over the internet, run a tunnel next to it, for example
`cloudflared tunnel --url http://localhost:8000`, and set STUDIO_PASSWORD.
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Dubbing Studio web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from .app import app

    print(
        f"\n  Dubbing Studio → http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}\n"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
