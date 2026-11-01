"""Kubernetes Agent Sandbox provider backed by ``agent_sandbox_rl`` ``SandboxFleet``."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from .base import ExecResult, Sandbox
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    from kubernetes.stream import stream as k8s_stream
except ImportError:
    k8s_client = None  # type: ignore[assignment]
    k8s_config = None  # type: ignore[assignment]
    k8s_stream = None  # type: ignore[assignment]

try:
    from agent_sandbox_rl import ClusterConfig, FleetConfig, SandboxFleet, Task
    import agent_sandbox_rl.resources as asrl_resources
except ImportError:
    ClusterConfig = None  # type: ignore[assignment]
    FleetConfig = None  # type: ignore[assignment]
    SandboxFleet = None  # type: ignore[assignment]
    Task = None  # type: ignore[assignment]
    asrl_resources = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

RUNTIME_CONTAINER_NAME = "agent-runtime"
_PWD_SENTINEL = "__UNI_AGENT_PWD__"

_FLEET_LOCK = threading.Lock()
_GLOBAL_FLEET: Any = None
_TEMPLATES_PATCHED = False


def _read_in_cluster_namespace() -> str | None:
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", encoding="utf-8") as f:
            ns = f.read().strip()
            if ns:
                return ns
    except OSError:
        pass
    return None


def _resolve_default_namespace(namespace: str | None = None) -> str:
    if namespace and namespace.strip():
        return namespace.strip()
    for env_key in ("SANDBOX_NAMESPACE", "K8S_NAMESPACE", "POD_NAMESPACE"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    return _read_in_cluster_namespace() or "default"


def patch_agent_sandbox_rl_templates() -> None:
    """Patches generated SandboxTemplates for GKE ownership annotations and node tolerations."""
    global _TEMPLATES_PATCHED
    if asrl_resources is None or _TEMPLATES_PATCHED:
        return
    if not hasattr(asrl_resources.Resources, "_orig_template_manifest") and hasattr(
        asrl_resources.Resources, "_template_manifest"
    ):
        asrl_resources.Resources._orig_template_manifest = asrl_resources.Resources._template_manifest

        def _patched_template_manifest(self: Any, image: str, template_name: str, template: Any) -> dict[str, Any]:
            manifest = self._orig_template_manifest(image, template_name, template)
            spec = manifest["spec"]
            spec["networkPolicyManagement"] = "Unmanaged"
            pod_meta = spec["podTemplate"].setdefault("metadata", {})
            pod_annotations = pod_meta.setdefault("annotations", {})
            ns = _resolve_default_namespace(getattr(self, "namespace", None))
            caller_ns = _read_in_cluster_namespace() or ns
            co_owners = ",".join(
                dict.fromkeys(
                    [
                        f"system:serviceaccount:{caller_ns}:default",
                        f"system:serviceaccount:{ns}:default",
                        f"system:serviceaccount:{ns}:xpk-sa",
                    ]
                )
            )
            pod_annotations.setdefault("ownership.tpu.google.com/creator", f"system:serviceaccount:{caller_ns}:default")
            pod_annotations.setdefault("ownership.tpu.google.com/co-owners", co_owners)
            pod_spec = spec["podTemplate"]["spec"]
            pod_spec["automountServiceAccountToken"] = False
            pod_spec["shareProcessNamespace"] = False
            node_sel = pod_spec.get("nodeSelector") or {}
            if node_sel:
                tolerations = pod_spec.setdefault("tolerations", [])
                for k, v in node_sel.items():
                    if not any(t.get("key") == k and t.get("value") == str(v) for t in tolerations):
                        tolerations.append({"key": k, "operator": "Equal", "value": str(v), "effect": "NoSchedule"})
            containers = pod_spec["containers"]
            if containers:
                containers[0]["name"] = RUNTIME_CONTAINER_NAME
            return manifest

        asrl_resources.Resources._template_manifest = _patched_template_manifest
        _TEMPLATES_PATCHED = True


def ensure_tasks_in_fleet_plan(fleet: Any, tasks: list[Any], *, warm: bool = True) -> None:
    """Registers ``tasks`` in ``fleet.plan_`` and creates warm pools for new images."""
    with _FLEET_LOCK:
        new_images = [
            t.image
            for t in tasks
            if getattr(t, "image", None) and (fleet.plan_ is None or fleet.plan_.for_image(t.image) is None)
        ]
        new_images = list(dict.fromkeys(new_images))
        if new_images or fleet.plan_ is None:
            existing = list(fleet.tasks) if getattr(fleet, "tasks", None) else []
            fleet.load_tasks(existing + tasks)
            fleet.plan()
            if warm and new_images and hasattr(fleet, "warm_images"):
                replicas = getattr(getattr(fleet, "config", None), "max_warmpool_size", None) or 4
                fleet.warm_images(new_images, replicas_override=replicas, wait=False)


def _acquire_from_fleet(
    image: str,
    *,
    namespace: str,
    node_selector: dict[str, str] | None,
) -> Any:
    global _GLOBAL_FLEET
    if ClusterConfig is None or FleetConfig is None or SandboxFleet is None or Task is None:
        raise ImportError("K8sAgentSandbox requires 'agent_sandbox_rl'.")
    with _FLEET_LOCK:
        if _GLOBAL_FLEET is None:
            patch_agent_sandbox_rl_templates()
            adopt_existing = os.getenv("SANDBOX_ADOPT_EXISTING", "0").lower() in ("1", "true", "yes")
            run_id = os.getenv("SANDBOX_RUN_ID", "uni-agent-shared").strip() or None
            cfg = FleetConfig(
                clusters=[
                    ClusterConfig(
                        name="default",
                        namespace=namespace,
                        node_selector=node_selector,
                        in_cluster=os.getenv("KUBERNETES_SERVICE_HOST") is not None,
                    )
                ],
                max_concurrent=128,
                window_size=8,
                max_warmpool_size=8,
                warm_per_task=True,
                adopt_existing=adopt_existing,
                run_id=run_id,
                install_teardown_hooks=True,
            )
            _GLOBAL_FLEET = SandboxFleet(cfg)
            _GLOBAL_FLEET._install_teardown_hooks()
        fleet = _GLOBAL_FLEET

    task_obj = Task(id=image, image=image, metadata={"ds": {"docker_image": image}})
    ensure_tasks_in_fleet_plan(fleet, [task_obj])
    for attempt in range(5):
        try:
            return fleet.acquire(task_obj)
        except Exception as exc:
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
            else:
                raise exc


class _K8sStatefulShell:
    """Lightweight shell wrapper that persists ``cwd`` and ``env`` without requiring ``tmux``."""

    def __init__(self, sandbox: K8sAgentSandbox, *, cwd: str | None = None, env: dict[str, str] | None = None) -> None:
        self._sandbox = sandbox
        self.cwd = cwd or "/testbed"
        self.env = dict(env or {})
        self._env_file = f"/tmp/.uni_agent_env_{uuid.uuid4().hex[:8]}"
        self._initialized = False

    async def run(self, command: str, *, timeout: float | None = 120.0) -> ExecResult:
        if not self._initialized:
            exports = "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in self.env.items())
            await self._sandbox.exec_shell(f"cat <<'__EOF__' > {self._env_file}\n{exports}\n__EOF__", timeout=30.0)
            self._initialized = True
        wrapped = (
            'if [ -d /opt/miniconda3/envs/testbed/bin ]; then export PATH="/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:${PATH}"; fi\n'
            'if [ -d /testbed/.venv/bin ]; then export PATH="/testbed/.venv/bin:${PATH}"; fi\n'
            f"[ -f {self._env_file} ] && . {self._env_file} >/dev/null 2>&1 || true\n"
            f"cd {shlex.quote(self.cwd)} 2>/dev/null || cd /\n"
            f"{command}\n"
            "__ec=$?\n"
            f'printf "\\n{_PWD_SENTINEL}:%s\\n" "$(pwd)"\n'
            f"export -p > {self._env_file} 2>/dev/null || true\n"
            "exit $__ec"
        )
        res = await self._sandbox.exec(["bash", "-c", wrapped], timeout=timeout, workdir=self.cwd, env=self.env)
        m = re.search(rf"\n?{re.escape(_PWD_SENTINEL)}:([^\r\n]+)\r?\n?$", res.stdout or "")
        stdout = res.stdout or ""
        if m:
            self.cwd = m.group(1).strip() or self.cwd
            stdout = stdout[: m.start()]
        return ExecResult(exit_code=res.exit_code, stdout=stdout, stderr=res.stderr)

    async def close(self) -> None:
        if self._initialized:
            await self._sandbox.exec(["rm", "-f", self._env_file], timeout=10.0)
            self._initialized = False


@register_sandbox("k8s_agent_sandbox")
class K8sAgentSandbox(Sandbox):
    """Kubernetes sandbox provider backed by ``agent_sandbox_rl`` warm pools."""

    supports_shell: ClassVar[bool] = True

    def __init__(
        self,
        *,
        image: str = "python:3.12",
        runtime_timeout: float = 1800.0,
        namespace: str | None = None,
        node_selector: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        image_prefix = os.getenv("SANDBOX_IMAGE_PREFIX", "").strip().rstrip("/")
        if image_prefix and "/" in image and image.startswith("namanjain12/"):
            image = f"{image_prefix}/{image.split('/', 1)[1]}"
        self.image = image
        self.runtime_timeout = float(runtime_timeout)
        self.namespace = _resolve_default_namespace(namespace)
        if node_selector is None:
            k, v = os.getenv("NODE_SELECTOR_KEY"), os.getenv("NODE_SELECTOR_VAL")
            self.node_selector = {k: v} if (k and v) else None
        else:
            self.node_selector = dict(node_selector)
        self._handle: Any = None
        self._pod_name: str | None = None
        self._started = False

    @classmethod
    def from_config(cls, config: SandboxConfig) -> K8sAgentSandbox:
        return cls(image=config.image, runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    def _get_core_v1_api(self) -> Any:
        if self._handle is not None and getattr(self._handle, "_cluster", None) is not None:
            return self._handle._cluster.exec_core_api()
        if k8s_client is None or k8s_config is None:
            raise ImportError("The 'kubernetes' package is required for K8sAgentSandbox.")
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        return k8s_client.CoreV1Api()

    async def start(self) -> None:
        if self._started:
            return
        self._handle = await asyncio.to_thread(
            _acquire_from_fleet,
            self.image,
            namespace=self.namespace,
            node_selector=self.node_selector,
        )
        self._pod_name = getattr(self._handle, "pod_name", None)
        if getattr(self._handle, "_cluster", None) is not None:
            self.namespace = self._handle._cluster.namespace
        self._started = True

    async def stop(self) -> None:
        self._started = False
        handle, self._handle = self._handle, None
        self._pod_name = None
        if handle is not None and _GLOBAL_FLEET is not None:
            await asyncio.to_thread(_GLOBAL_FLEET.release, handle)

    async def is_alive(self) -> bool:
        return self._started and self._handle is not None

    async def open_shell(self, *, cwd: str | None = None, env: dict[str, str] | None = None) -> _K8sStatefulShell:
        return _K8sStatefulShell(self, cwd=cwd, env=env)

    def _exec_in_pod_sync(
        self,
        argv: list[str],
        timeout: float,
        workdir: str | None,
        env: dict[str, str] | None,
    ) -> ExecResult:
        if self._pod_name is None or k8s_stream is None:
            raise RuntimeError("K8sAgentSandbox is not started or kubernetes.stream is unavailable.")
        inner_cmd = " ".join(shlex.quote(a) for a in argv)
        parts = [f"export {k}={shlex.quote(str(v))}" for k, v in (env or {}).items()]
        if workdir:
            parts.append(f"cd {shlex.quote(workdir)}")
        full_cmd = (" && ".join(parts) + f" && exec {inner_cmd}") if parts else inner_cmd
        timeout_int = max(1, int(timeout))
        wrapped_argv = ["/bin/sh", "-c", f"timeout {timeout_int}s /bin/sh -c {shlex.quote(full_cmd)}"]

        api = self._get_core_v1_api()
        resp = k8s_stream(
            api.connect_get_namespaced_pod_exec,
            self._pod_name,
            self.namespace,
            command=wrapped_argv,
            container=RUNTIME_CONTAINER_NAME,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        deadline = time.monotonic() + timeout + 10.0
        try:
            while resp.is_open():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    resp.close()
                    raise TimeoutError(f"Pod exec exceeded {timeout}s on {self._pod_name}/{RUNTIME_CONTAINER_NAME}")
                resp.update(timeout=min(1.0, remaining))
                if resp.peek_stdout():
                    stdout_chunks.append(resp.read_stdout())
                if resp.peek_stderr():
                    stderr_chunks.append(resp.read_stderr())
            rc = resp.returncode
        finally:
            try:
                resp.close()
            except Exception:
                pass

        exit_code = int(rc) if rc is not None else 0
        stdout_str = "".join(stdout_chunks)
        stderr_str = "".join(stderr_chunks)
        if exit_code == 124:
            raise TimeoutError(stdout_str or stderr_str or f"Command timed out after {timeout}s")
        return ExecResult(exit_code=exit_code, stdout=stdout_str, stderr=stderr_str)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if not self._started or self._handle is None:
            raise RuntimeError("K8sAgentSandbox is not started.")
        eff_timeout = max(float(timeout or 120.0), 1.0)
        return await asyncio.wait_for(
            asyncio.to_thread(self._exec_in_pod_sync, argv, eff_timeout, workdir, env),
            timeout=eff_timeout + 15.0,
        )
