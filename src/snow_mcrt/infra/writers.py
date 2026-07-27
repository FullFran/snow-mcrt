"""Result output: a CSV of the numbers and a JSON manifest of what produced them.

The split matters. "The plot looks right" is not a result; "here are the
numbers, and here is the manifest that produced them" is. A committed CSV plus
its manifest means a figure can be regenerated without re-running any physics,
and two runs can be compared parameter by parameter instead of by eye.

The manifest records **every** input, seed included. If a field is absent from
it, that field could not have influenced the result -- and if that ever stops
being true, reproducibility is gone and nothing will announce it.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from snow_mcrt.application.spectral_albedo import SpectralAlbedoResult


def _git_commit() -> str | None:
    """Current commit, or ``None`` outside a repository.

    Recorded so a result can be traced to the code that produced it. A run
    made with uncommitted changes is marked, because "the version in git" is
    then a lie.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.SubprocessError, OSError):
        return None


def _jsonable(value: Any) -> Any:
    """Convert numpy and dataclass values into something JSON can hold."""
    if isinstance(value, np.ndarray):
        if value.size > 8:
            return {
                "min": float(value.min()),
                "max": float(value.max()),
                "count": int(value.size),
            }
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def write_csv(result: SpectralAlbedoResult, path: str | Path) -> Path:
    """Write the result columns as CSV.

    Args:
        result: The curve to write.
        path: Destination. Parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = result.columns()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns.keys())
        for row in zip(*columns.values()):
            writer.writerow(f"{value:.10g}" for value in row)
    return path


def write_manifest(result: SpectralAlbedoResult, path: str | Path) -> Path:
    """Write every parameter that produced the result, as JSON.

    Args:
        result: The curve whose provenance is being recorded.
        path: Destination. Parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {k: _jsonable(v) for k, v in asdict(result.config).items()}
    manifest = {
        "label": result.config.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "optical_constants": result.dataset_name,
        "config": config,
        "summary": {
            "wavelength_nm": [
                float(result.wavelength_nm.min()),
                float(result.wavelength_nm.max()),
            ],
            "n_points": int(result.wavelength_nm.size),
            "albedo_min": float(result.albedo.min()),
            "albedo_max": float(result.albedo.max()),
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def write_result(
    result: SpectralAlbedoResult, directory: str | Path, stem: str | None = None
) -> tuple[Path, Path]:
    """Write both the CSV and its manifest, named consistently.

    Args:
        result: The curve to persist.
        directory: Destination directory.
        stem: Base filename. Defaults to the config label.

    Returns:
        ``(csv_path, manifest_path)``.
    """
    stem = stem or result.config.label
    directory = Path(directory)
    return (
        write_csv(result, directory / f"{stem}.csv"),
        write_manifest(result, directory / f"{stem}.manifest.json"),
    )
