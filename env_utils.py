from __future__ import annotations

import os
from pathlib import Path


def parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = parse_env_value(value)


def load_default_env_file(env_override_var: str | None = None) -> None:
    if env_override_var:
        env_override = os.environ.get(env_override_var)
        if env_override:
            load_env_file(Path(env_override))
            return

    repo_env = Path(__file__).resolve().parent / ".env"
    load_env_file(repo_env)
