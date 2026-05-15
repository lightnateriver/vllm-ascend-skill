from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/mnt/sfs_turbo/codes/ascend/vllm-ascend-skill")
FC_SCRIPT = REPO_ROOT / "vllm-multimodal-evaluator" / "scripts" / "fc_test.py"
REGRESSION_SCRIPT = (
    REPO_ROOT
    / "vllm-multimodal-precision-testing"
    / "scripts"
    / "run_full_regression.py"
)
STANDARD_RETEST_SCRIPT = (
    REPO_ROOT
    / "vllm-multimodal-precision-testing"
    / "scripts"
    / "run_standard_retest.py"
)


def _extract_json_suffix(stdout: str) -> dict:
    lines = stdout.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("{"):
            candidate = "\n".join(lines[idx:])
            return json.loads(candidate)
    raise AssertionError("No JSON object found in stdout")


def test_fc_help_advertises_json_flag() -> None:
    proc = subprocess.run(
        [sys.executable, str(FC_SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--json" in proc.stdout


def test_fc_json_contract_allows_machine_parsing() -> None:
    sample = """FAIL | FC-001 [基础功能] 必选参数 ['date'] 缺失，实际参数：['city']\nPASS | FC-002 [基础功能] 调用 get_weather({'city': '上海'})\n\n=== 汇总: 12/15 通过 ===\n{\n  "summary": {\n    "passed": 12,\n    "failed": 3,\n    "total": 15,\n    "pass_rate": 0.8,\n    "engineering_error_cases": [],\n    "model_limitation_cases": ["FC-001", "FC-006", "FC-009"]\n  },\n  "results": []\n}\n"""
    parsed = _extract_json_suffix(sample)
    assert parsed["summary"]["total"] == 15
    assert parsed["summary"]["failed"] == 3
    assert parsed["summary"]["model_limitation_cases"] == [
        "FC-001",
        "FC-006",
        "FC-009",
    ]


def test_regression_help_advertises_l05_controls() -> None:
    proc = subprocess.run(
        [sys.executable, str(REGRESSION_SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--l05-dataset-dir" in proc.stdout
    assert "--skip-l05" in proc.stdout
    assert "--media-modes" in proc.stdout
    assert "--auto-start-media-server" in proc.stdout
    assert "--no-auto-start-media-server" in proc.stdout
    assert "--output-root" in proc.stdout
    assert "--run-name" in proc.stdout


def test_standard_retest_help_advertises_orchestration_flags() -> None:
    proc = subprocess.run(
        [sys.executable, str(STANDARD_RETEST_SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--skip-capability" in proc.stdout
    assert "--skip-precision" in proc.stdout
    assert "--output-root" in proc.stdout
    assert "--run-name" in proc.stdout
    assert "--no-auto-start-media-server" in proc.stdout


def test_regression_summary_contract_shape() -> None:
    sample = {
        "run_name": "full_regression_20260515_120000",
        "artifact_root": "/tmp/full_regression_20260515_120000",
        "requested_media_modes": ["base64", "local_path", "http"],
        "mode_count": 3,
        "modes": {
            "base64": {
                "media_mode": "base64",
                "steps": {
                    "l0": {
                        "passed": True,
                        "returncode": 0,
                        "parsed": {"summary": {"passed": 10}},
                        "artifacts": {"step_dir": "/tmp/l0"},
                    },
                    "l05": {
                        "passed": False,
                        "returncode": 1,
                        "parsed": {
                            "total": 40,
                            "correct": 23,
                            "wrong": 17,
                            "unknown": 0,
                            "timeout": 0,
                            "accuracy": 0.575,
                        },
                        "artifacts": {"step_dir": "/tmp/l05"},
                    },
                },
                "overall_pass": False,
                "precision_summary": {
                    "l0": {"passed": 10, "failed": 0, "total": 10},
                    "l05": {
                        "total": 40,
                        "correct": 23,
                        "wrong": 17,
                        "unknown": 0,
                        "timeout": 0,
                        "accuracy": 0.575,
                    },
                    "mme": {"exact_acc": 80.74, "unknown": 7},
                    "mmbench": {"overall_acc": 82.81, "z_fallback": 15},
                },
                "engineering_errors": [],
                "model_limitations": ["13", "14"],
                "format_or_protocol_issues": [],
                "artifact_paths": {},
                "final_verdict": {
                    "deployment_ready_for_precision": True,
                    "precision_status": "partial_or_fail",
                    "notes": [],
                },
            }
        },
        "mode_comparison": {
            "l0": {
                "base64": {"passed": 10, "failed": 0, "total": 10},
                "local_path": {"passed": 10, "failed": 0, "total": 10},
                "http": {"passed": 10, "failed": 0, "total": 10},
            },
            "mme": {
                "base64": {"exact_acc": 80.74, "unknown": 7},
                "local_path": {"exact_acc": 80.70, "unknown": 7},
                "http": {"exact_acc": 80.66, "unknown": 7},
            },
        },
        "overall_pass": False,
        "downloads": [],
        "summary_paths": {
            "json": "/tmp/full_regression_20260515_120000/summary.json",
            "md": "/tmp/full_regression_20260515_120000/summary.md",
        },
        "final_verdict": {"all_modes_passed": False, "required_modes": ["base64", "local_path", "http"], "notes": []},
    }
    base64 = sample["modes"]["base64"]
    assert sample["run_name"] == "full_regression_20260515_120000"
    assert sample["artifact_root"] == "/tmp/full_regression_20260515_120000"
    assert sample["mode_count"] == 3
    assert sample["requested_media_modes"] == ["base64", "local_path", "http"]
    assert base64["precision_summary"]["l0"]["total"] == 10
    assert base64["precision_summary"]["l05"]["accuracy"] == 0.575
    assert base64["precision_summary"]["mme"]["unknown"] == 7
    assert base64["precision_summary"]["mmbench"]["z_fallback"] == 15
    assert base64["steps"]["l0"]["artifacts"]["step_dir"] == "/tmp/l0"
    assert sample["mode_comparison"]["l0"]["http"]["total"] == 10
    assert sample["summary_paths"]["json"].endswith("/summary.json")
    assert sample["final_verdict"]["all_modes_passed"] is False
