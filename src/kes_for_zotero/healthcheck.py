from __future__ import annotations

import importlib
import importlib.metadata
import json
from dataclasses import dataclass
from urllib import error, request

from kes_for_zotero.config import AppConfig


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_self_check(config: AppConfig) -> tuple[bool, list[CheckResult]]:
    results: list[CheckResult] = []

    storage_ok = config.storage_root.exists() and config.storage_root.is_dir()
    results.append(
        CheckResult(
            name="storage_root",
            ok=storage_ok,
            detail=f"{config.storage_root} ({'ok' if storage_ok else 'missing or not a directory'})",
        )
    )

    marker_base_url = config.marker.ollama_base_url.rstrip("/")
    marker_tags_ok, marker_tags_detail, marker_models = _fetch_ollama_models(marker_base_url)
    results.append(
        CheckResult(
            name="ollama_connection",
            ok=marker_tags_ok,
            detail=marker_tags_detail,
        )
    )

    model_names_to_check: list[tuple[str, str]] = [("marker_model", config.marker.ollama_model)]
    if config.vision.enabled:
        model_names_to_check.append(("vision_model", config.vision.model))

    checked_names: set[str] = set()
    for check_name, model_name in model_names_to_check:
        if model_name in checked_names:
            continue
        checked_names.add(model_name)
        model_ok, model_detail = _check_model_name(model_name, marker_models, marker_tags_ok)
        results.append(CheckResult(name=check_name, ok=model_ok, detail=model_detail))

    runtime_ok, runtime_detail = _check_marker_runtime_dependencies()
    results.append(CheckResult(name="marker_runtime", ok=runtime_ok, detail=runtime_detail))

    return all(item.ok for item in results), results


def _fetch_ollama_models(base_url: str) -> tuple[bool, str, set[str]]:
    endpoint = base_url + "/api/tags"
    req = request.Request(endpoint, method="GET")

    try:
        with request.urlopen(req, timeout=8) as response:
            body = response.read().decode("utf-8")
    except error.URLError as exc:
        return False, f"{endpoint} unreachable: {exc}", set()

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return False, f"{endpoint} returned invalid JSON: {exc}", set()

    models = payload.get("models", [])
    names = {
        str(item.get("name") or item.get("model") or "").strip()
        for item in models
        if isinstance(item, dict)
    }
    names = {name for name in names if name}
    return True, f"{endpoint} reachable, {len(names)} model(s) listed", names


def _check_model_name(model_name: str, available_models: set[str], connection_ok: bool) -> tuple[bool, str]:
    if not connection_ok:
        return False, f"cannot verify '{model_name}' because Ollama is unreachable"

    # Accept exact match and shorthand form without explicit tag.
    if model_name in available_models:
        return True, f"'{model_name}' found"

    base_name = model_name.split(":", 1)[0]
    if any(candidate.split(":", 1)[0] == base_name for candidate in available_models):
        return True, f"'{model_name}' matched by base name '{base_name}'"

    return False, f"'{model_name}' not found in Ollama tags"


def _check_marker_runtime_dependencies() -> tuple[bool, str]:
    versions: dict[str, str] = {}
    for module_name in ("torch", "torchvision", "transformers", "marker-pdf"):
        try:
            versions[module_name] = importlib.metadata.version(module_name)
        except importlib.metadata.PackageNotFoundError:
            versions[module_name] = "not-installed"

    for module_name in ("torch", "torchvision", "transformers", "marker"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name + "_imported"] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            return False, f"failed to import {module_name}: {exc}; versions={versions}"

    try:
        importlib.import_module("marker.config.parser")
        importlib.import_module("marker.converters.pdf")
    except Exception as exc:
        return (
            False,
            "marker runtime import failed. "
            f"versions={versions}. "
            f"error={exc}",
        )

    return True, f"marker runtime import ok, versions={versions}"
