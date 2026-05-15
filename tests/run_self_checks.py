from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _run_test_functions(module) -> tuple[int, int]:
    total = 0
    failed = 0
    for name in dir(module):
        if not name.startswith("test_"):
            continue
        fn = getattr(module, name)
        if not callable(fn):
            continue
        total += 1
        try:
            fn()
            print(f"PASS {module.__name__}.{name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {module.__name__}.{name}")
            traceback.print_exc()
    return total, failed


def main() -> int:
    test_files = [
        TEST_DIR / "test_output_contracts.py",
        TEST_DIR / "test_transport_consistency.py",
    ]
    total = 0
    failed = 0
    for test_file in test_files:
        module = _load_module(test_file)
        sub_total, sub_failed = _run_test_functions(module)
        total += sub_total
        failed += sub_failed
    print(f"SUMMARY total={total} failed={failed} passed={total - failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
