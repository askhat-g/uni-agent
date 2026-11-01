"""Unit tests for K8sAgentSandbox and R2E-Gym / SWE task config resolution."""

from __future__ import annotations

import asyncio
import json

from uni_agent.sandbox import SandboxConfig, build_sandbox
from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.k8s_agent_sandbox import K8sAgentSandbox, _K8sStatefulShell
from uni_agent.tasks import TaskConfigResolver, get_task
from uni_agent.tasks.r2e_gym import R2EGymTask, compute_reward, parse_log_pytest


def test_k8s_agent_sandbox_registered():
    cfg = SandboxConfig(provider="k8s_agent_sandbox", image="swerebench/test:latest", runtime_timeout=600.0)
    sb = build_sandbox(cfg)
    assert isinstance(sb, K8sAgentSandbox)
    assert sb.image == "swerebench/test:latest"


def test_stateful_shell_persists_cwd_and_conda_path():
    class _DummyK8sSandbox(K8sAgentSandbox):
        def __init__(self):
            super().__init__(image="dummy")
            self.last_cmd = ""

        async def _exec(self, argv, *, timeout=None, workdir=None, env=None):
            self.last_cmd = " ".join(argv)
            if "cd /testbed/sub" in self.last_cmd:
                return ExecResult(exit_code=0, stdout="ok\n__UNI_AGENT_PWD__:/testbed/sub\n", stderr="")
            return ExecResult(exit_code=0, stdout="", stderr="")

    sb = _DummyK8sSandbox()
    shell = _K8sStatefulShell(sb, cwd="/testbed", env={"FOO": "bar"})
    res = asyncio.run(shell.run("cd /testbed/sub && echo ok"))
    assert res.exit_code == 0
    assert res.stdout == "ok"
    assert shell.cwd == "/testbed/sub"
    assert "/opt/miniconda3/envs/testbed/bin" in sb.last_cmd


def test_task_config_react_resolves_r2e_gym_and_swe():
    resolver = TaskConfigResolver.from_file("examples/quickstart/training/task_config_react.yaml")
    for task_name in ("r2e_gym", "swe_rebench", "swe_bench"):
        resolved = resolver.resolve(
            {
                "name": task_name,
                "sandbox": {"image": "r2egym/test:latest"},
                "metadata": {"problem_statement": "Fix bug"},
            }
        )
        assert resolved["sandbox"]["provider"] == "k8s_agent_sandbox"
        assert resolved["agent"]["name"] == "react"
        if task_name == "r2e_gym":
            assert isinstance(get_task(resolved), R2EGymTask)


def test_r2e_gym_compute_reward():
    class _EvalSandbox(K8sAgentSandbox):
        async def _exec(self, argv, *, timeout=None, workdir=None, env=None):
            out = "=== short test summary info ===\nPASSED test_mod.py::TestCls::test_one\n"
            return ExecResult(exit_code=0, stdout=out, stderr="")

    parsed = parse_log_pytest("=== short test summary info ===\nPASSED test_mod.py::TestCls::test_one\n")
    assert parsed == {"TestCls.test_one": "PASSED"}
    res = asyncio.run(
        compute_reward(
            {"instance_id": "repo__1", "expected_output_json": json.dumps({"TestCls.test_one": "PASSED"})},
            _EvalSandbox(image="dummy"),
        )
    )
    assert res["resolved"] is True
