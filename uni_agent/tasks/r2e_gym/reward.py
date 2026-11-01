"""R2E-Gym eval/reward (ported from pre-#83 uni_agent.reward.r2e_gym)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...sandbox import SandboxBackend

logger = logging.getLogger(__name__)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m|\u001b\[\d+m|\r")


def parse_log_pytest(log: str | None) -> dict[str, str]:
    """Parse R2E-Gym pytest ``short test summary info`` output into ``{test_name: status}``."""
    if not log:
        return {}
    clean_log = _ANSI_ESCAPE_RE.sub("", log)
    test_status_map: dict[str, str] = {}
    if "short test summary info" not in clean_log:
        return test_status_map
    summary = clean_log.split("short test summary info", 1)[1].strip()
    for line in summary.split("\n"):
        if "PASSED" in line:
            parts = line.split("::")
            if len(parts) > 1:
                test_status_map[".".join(parts[1:]).strip()] = "PASSED"
        elif "FAILED" in line:
            parts = line.split("::")
            if len(parts) > 1:
                test_status_map[".".join(parts[1:]).split(" - ")[0].strip()] = "FAILED"
        elif "ERROR" in line:
            parts = line.split("::")
            test_name = ".".join(parts[1:]) if len(parts) > 1 else line
            test_status_map[test_name.split(" - ")[0].strip()] = "ERROR"
    return test_status_map


def decolor_dict_keys(mapping: dict[str, str]) -> dict[str, str]:
    return {_ANSI_ESCAPE_RE.sub("", str(k)): v for k, v in mapping.items()}


def _get_eval_report(metadata: dict[str, Any], eval_output: str) -> dict[str, Any]:
    parsed_status = decolor_dict_keys(parse_log_pytest(eval_output))
    raw_expected = metadata.get("expected_output_json", "{}")
    expected_obj = json.loads(raw_expected) if isinstance(raw_expected, str) else dict(raw_expected or {})
    expected_status = decolor_dict_keys(expected_obj)

    parsed_norm = {k.split(" - ")[0]: parsed_status[k] for k in sorted(parsed_status)}
    expected_norm = {k.split(" - ")[0]: expected_status[k] for k in sorted(expected_status)}
    resolved = bool(expected_norm) and len(parsed_norm) == len(expected_norm) and all(
        k in expected_norm and parsed_norm[k] == expected_norm[k] for k in parsed_norm if k
    )
    return {
        "resolved": resolved,
        "found_eval_status": bool(parsed_status),
        "test_status": {"parsed_status": parsed_norm, "expected_status": expected_norm},
    }


async def compute_reward(metadata: dict[str, Any], sandbox: SandboxBackend, eval_timeout: float = 300.0) -> dict[str, Any]:
    """Restore hidden ``/r2e_tests`` and run ``run_tests.sh`` inside the sandbox."""
    instance_id = metadata.get("instance_id", "?")
    restore_and_test = (
        "export PATH=/testbed/.venv/bin:/root/.venv/bin:/root/.local/bin:$PATH\n"
        "if [ -d /root/.r2e_eval/r2e_tests ]; then "
        "rm -rf /root/r2e_tests /testbed/r2e_tests && "
        "cp -a /root/.r2e_eval/r2e_tests /root/r2e_tests && "
        "ln -s /root/r2e_tests /testbed/r2e_tests; fi\n"
        "if [ -f /root/.r2e_eval/run_tests.sh ]; then bash /root/.r2e_eval/run_tests.sh; "
        "elif [ -f /root/run_tests.sh ]; then bash /root/run_tests.sh; "
        "elif [ -f /run_tests.sh ]; then bash /run_tests.sh; "
        "else echo 'run_tests.sh not found' >&2; exit 1; fi"
    )
    t0 = time.perf_counter()
    resp = await sandbox.exec_shell(f"{restore_and_test} 2>&1", workdir="/testbed", timeout=eval_timeout)
    elapsed = time.perf_counter() - t0
    output = _ANSI_ESCAPE_RE.sub("", resp.stdout or "")
    eval_report = _get_eval_report(metadata, output)
    logger.info("r2e_gym reward for %s: resolved=%s (%.1fs)", instance_id, eval_report["resolved"], elapsed)
    return {
        "eval_completed": resp.exit_code == 0,
        "eval_execution_time": elapsed,
        "eval_report": eval_report,
        "resolved": eval_report["resolved"],
    }
