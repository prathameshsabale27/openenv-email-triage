"""FastAPI app for the email triage OpenEnv environment."""

from __future__ import annotations

import argparse

import uvicorn
from openenv.core.env_server.http_server import create_app

from email_triage_env.models import TriageAction, TriageObservation
from server.email_triage_environment import OpenEnvEmailTriageEnvironment


app = create_app(
    OpenEnvEmailTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="email-triage-openenv",
    max_concurrent_envs=2,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


def _cli_main() -> None:
    """Parse command-line options and launch the server."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)


if __name__ == "__main__":
    # main()
    _cli_main()
