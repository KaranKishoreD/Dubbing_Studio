"""
Every module imports, and every name imported from the `pipeline` package
exists — including imports written inside functions, which a plain
`import pipeline` would not exercise.
"""

import ast
import importlib
import importlib.util
import pathlib
import pkgutil

import pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Run inside their own virtual environments (LatentSync, Seed-VC, NeMo).
_EXTERNAL_WORKERS = ("latentsync_worker", "seedvc_worker", "syncnet_worker")


def test_every_module_imports():
    for info in pkgutil.walk_packages(pipeline.__path__, "pipeline."):
        if info.name.endswith(_EXTERNAL_WORKERS):
            continue
        importlib.import_module(info.name)


def test_every_pipeline_import_resolves():
    problems = []
    files = [
        *(ROOT / "pipeline").rglob("*.py"),
        *(ROOT / "server").glob("*.py"),
    ]
    for path in files:
        if path.stem in _EXTERNAL_WORKERS:
            continue
        package = ".".join(path.relative_to(ROOT).parent.parts)
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                parts = package.split(".")
                target = ".".join(parts[: len(parts) - node.level + 1])
                target += f".{node.module}" if node.module else ""
            elif node.module and node.module.split(".")[0] in ("pipeline", "server"):
                target = node.module
            else:
                continue
            try:
                module = importlib.import_module(target)
            except ImportError as exc:
                problems.append(f"{path.name}:{node.lineno} cannot import {target} ({exc})")
                continue
            for alias in node.names:
                is_submodule = (
                    hasattr(module, "__path__")
                    and importlib.util.find_spec(f"{target}.{alias.name}") is not None
                )
                if not hasattr(module, alias.name) and not is_submodule:
                    problems.append(f"{path.name}:{node.lineno} {target} has no {alias.name}")
    assert not problems, "\n".join(problems)
