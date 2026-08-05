"""
Guarantees the project root (the parent of this `tests/` folder, where
the `app` package lives) is on sys.path before any test module is
imported.

This is redundant with `pyproject.toml`'s `[tool.pytest.ini_options]
pythonpath = ["."]` setting by design — conftest.py is *always*
auto-discovered by pytest before test collection starts, regardless of
which directory you invoke `pytest` from, whether the config file was
picked up, or stale `.pytest_cache` state. If you ever see
`ModuleNotFoundError: No module named 'app'` again despite this file
being present, the most likely cause is that this conftest.py itself
didn't get copied into your local project's `tests/` folder.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
