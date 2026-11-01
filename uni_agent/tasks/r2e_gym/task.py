"""R2E-Gym task (native framework loop)."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from ...sandbox import SandboxBackend
from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import compute_reward

logger = logging.getLogger(__name__)

_SETUP_R2E_WORKSPACE = """
mkdir -p /root/.local/bin /root/.r2e_eval
if [ -d /testbed/.venv ]; then
    ln -sfn /testbed/.venv /root/.venv
    find /testbed/.venv/bin -maxdepth 1 -type f -executable -exec ln -sf {} /root/.local/bin/ \\; 2>/dev/null || true
fi
for src in /testbed/run_tests.sh /run_tests.sh /root/run_tests.sh; do
    if [ -f "$src" ] && [ ! -f /root/.r2e_eval/run_tests.sh ]; then cp -a "$src" /root/.r2e_eval/run_tests.sh; fi
done
for src in /r2e_tests /testbed/r2e_tests /root/r2e_tests; do
    if [ -d "$src" ] && [ ! -L "$src" ] && [ ! -d /root/.r2e_eval/r2e_tests ]; then cp -a "$src" /root/.r2e_eval/r2e_tests; fi
done
rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests /run_tests.sh /root/run_tests.sh /testbed/run_tests.sh 2>/dev/null || true
git config --global --add safe.directory /testbed 2>/dev/null || true
git -C /testbed reset --hard 2>/dev/null || true
""".strip()


async def _prepare_workspace(sandbox: SandboxBackend) -> None:
    await sandbox.exec_shell(_SETUP_R2E_WORKSPACE, workdir="/testbed", timeout=120.0)


class R2EGymTaskConfig(TaskConfig):
    name: str = "r2e_gym"
    run_oracle_solution: bool = Field(default=False)
    eval_timeout: float = Field(default=300.0)


@register_task("r2e_gym")
class R2EGymTask(Task):
    name = "r2e_gym"
    config_model = R2EGymTaskConfig

    async def run(self) -> TaskResult:
        cfg: R2EGymTaskConfig = self.config  # type: ignore[assignment]
        sample: dict[str, Any] = dict(cfg.metadata) if isinstance(cfg.metadata, dict) else {}

        if cfg.sandbox.image is None and sample.get("docker_image"):
            cfg.sandbox.image = str(sample["docker_image"])

        async with self.build_sandbox() as sandbox:
            await _prepare_workspace(sandbox)

            if cfg.run_oracle_solution and sample.get("patch"):
                await sandbox.write_file("/tmp/gold_patch.patch", str(sample["patch"]))
                await sandbox.exec(["git", "apply", "--whitespace=fix", "/tmp/gold_patch.patch"], workdir="/testbed")
                finished = True
            else:
                agent = self.build_agent()
                agent_result = await agent.run(sandbox=sandbox, messages=cfg.prompt, workdir="/testbed")
                finished = agent_result.finished

            result = await compute_reward(sample, sandbox, eval_timeout=cfg.eval_timeout)
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                finished=finished,
                extra_info=result,
            )
