"""Runtime configuration for the generic Delta AI Chat package."""

from __future__ import annotations

import os
import subprocess


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} in the environment before starting Delta AI Chat.")
    return value


def authenticate_session(profile_name: str, *, required: bool = False) -> None:
    """Refresh an OCI session when region and tenancy are configured."""
    region = os.environ.get("DELTA_AI_REGION")
    tenancy = os.environ.get("DELTA_AI_TENANCY_NAME")
    if not region and not tenancy and not required:
        return
    if not region or not tenancy:
        raise RuntimeError(
            "Set DELTA_AI_REGION and DELTA_AI_TENANCY_NAME to authenticate or refresh a session."
        )
    subprocess.run(
        [
            "oci",
            "session",
            "authenticate",
            "--region",
            region,
            "--tenancy-name",
            tenancy,
            "--profile-name",
            profile_name,
        ],
        check=True,
    )
