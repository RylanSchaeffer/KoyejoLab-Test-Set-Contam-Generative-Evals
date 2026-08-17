"""Sandboxed execution scoring for MBPP-style code benchmarks (ICLR 2027 Phase 2).

Runs model-generated Python against a benchmark's `test_list` asserts in a
subprocess sandbox, mirroring how bigcode-evaluation-harness scores MBPP and
HumanEval, and returns a boolean the eval scripts can log exactly like a
`math_verify` score (checklist item 2.2: minimal schema change, per-problem
results stay in W&B run history).

Isolation is a correctness requirement, not a nicety -- this executes untrusted
model output (checklist 2.1). The design, in order of the guarantees it gives:

  * **Subprocess, never in-process.** The generated code runs under a fresh
    `python -I` (isolated mode: no site-packages injection from cwd, no user
    site, environment variables ignored), so nothing it does -- monkeypatching,
    `sys.exit`, segfaulting an extension -- can corrupt the evaluating process.
    `exec()` in-process is forbidden in this codebase for model output.
  * **Wall-clock timeout.** The subprocess runs in its own session
    (`start_new_session=True`); on timeout the whole process group is killed,
    so grandchildren cannot outlive the test.
  * **Resource limits** via `resource.setrlimit` in the child before exec:
    address space (default 1 GiB), CPU seconds (timeout + margin, a backstop in
    case the wall clock is evaded), file size (10 MiB, so a runaway write
    cannot fill the shared filesystem), and core dumps disabled.
  * **No network / no shell escapes**, best effort: a prelude executed before
    the untrusted code replaces `socket.socket`, `os.system`, `os.popen`, and
    the `subprocess` entry points with raisers, and stubs `builtins.input`.
    This is the same in-child guard style as bigcode's `reliability_guard`; it
    stops accidental and casual misuse, not a determined adversary -- which
    matches the threat model of scoring our own models' output on a research
    cluster.
  * **Scratch working directory.** Each execution gets a fresh temp dir as cwd,
    deleted afterwards, so stray files neither persist nor collide.

A crash, timeout, or nonzero exit scores 0 and never raises into the caller
(checklist 2.1: the failure path must not kill the eval run).

Protocol note (checklist 2.4): scoring is per-response, so pass@1 under greedy
decoding falls out of the existing single-sample eval loop exactly as it does
for MATH; pass@k comes from k seeded passes aggregated offline.
"""

import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Sequence

# The sentinel MBPP prompts use to close a solution (see src/data.py's
# MBPP_DOC_TO_TEXT). Everything a model generates after it is a new
# (hallucinated) turn, not part of the answer.
MBPP_STOP_SEQUENCE = "[DONE]"

# Markdown code fence, in case a model answers in chat style rather than the
# demonstrated [BEGIN]/[DONE] convention. Scoring a correct answer as wrong
# purely because of its wrapper would understate capability.
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MEMORY_LIMIT_MB = 1024
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Executed by the child *before* the untrusted code. Best-effort guard against
# network access and shell escapes; see module docstring for the threat model.
_SANDBOX_PRELUDE = """\
import builtins as _builtins
import os as _os
import socket as _socket


def _blocked(*_args, **_kwargs):
    raise RuntimeError("blocked by sandbox")


_socket.socket = _blocked
_socket.create_connection = _blocked
_os.system = _blocked
_os.popen = _blocked
_os.execv = _blocked
_os.execve = _blocked
_os.fork = _blocked
_os.forkpty = _blocked
_builtins.input = _blocked

import subprocess as _subprocess

_subprocess.Popen = _blocked
_subprocess.run = _blocked
_subprocess.call = _blocked
_subprocess.check_call = _blocked
_subprocess.check_output = _blocked
"""


@dataclass
class CodeExecutionResult:
    """Outcome of one sandboxed execution.

    `passed` is what gets logged; `reason` ("ok", "timeout", "exit code N:
    <stderr tail>", "no code extracted") exists so failures can be audited from
    run history without re-executing anything.
    """

    passed: bool
    reason: str


def extract_python_code(response_text: str) -> str | None:
    """Extract the Python program from a model response.

    Honours, in order:
      1. The MBPP [DONE] sentinel -- everything after the first one is a new
         hallucinated turn and is discarded (the prompt ends at "[BEGIN]", so
         the response opens directly with code).
      2. A markdown code fence, if present, in which case its first block is
         taken -- a chat-style model wraps its code rather than following the
         demonstrated sentinel convention.

    Returns None when nothing remains after unwrapping (an empty response, or
    text that is only whitespace) so that "no code" fails scoring cleanly and
    distinguishably from "code that fails its tests". Deliberately does NOT try
    to judge whether the remainder *is* code: garbage text simply fails to
    execute, which is the honest failure path.
    """
    text = response_text.split(MBPP_STOP_SEQUENCE, 1)[0]
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match is not None:
        text = fence_match.group(1)
    text = text.strip()
    if not text:
        return None
    return text


def _child_resource_limits(memory_limit_mb: int, cpu_seconds: int) -> None:
    """Applied in the child between fork and exec (runs in the child process)."""
    memory_bytes = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (_MAX_FILE_SIZE_BYTES, _MAX_FILE_SIZE_BYTES)
    )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_python_with_tests(
    code: str,
    test_list: Sequence[str],
    test_imports: Sequence[str] = (),
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
) -> CodeExecutionResult:
    """Execute `code` followed by `test_list` asserts in a sandboxed subprocess.

    Args:
        code: The (untrusted) Python program under test.
        test_list: Executable assert statements, MBPP's `test_list` column.
        test_imports: Import statements some MBPP tasks require before their
            asserts (`test_imports` column; usually empty).
        timeout_seconds: Wall-clock budget for the whole execution.
        memory_limit_mb: Address-space cap for the child.

    Returns:
        CodeExecutionResult; never raises for any behaviour of the code under
        test (syntax errors, exceptions, hangs, OOM, and asserts all come back
        as passed=False with a reason).
    """
    program = "\n".join([_SANDBOX_PRELUDE, code, "", *test_imports, *test_list, ""])
    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as scratch_dir:
        program_path = os.path.join(scratch_dir, "program.py")
        with open(program_path, "w") as f:
            f.write(program)

        process = subprocess.Popen(
            [sys.executable, "-I", program_path],
            cwd=scratch_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=lambda: _child_resource_limits(
                memory_limit_mb=memory_limit_mb,
                cpu_seconds=int(timeout_seconds) + 5,
            ),
        )
        try:
            _, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            # Kill the whole session so grandchildren die with the child.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return CodeExecutionResult(passed=False, reason="timeout")

    if process.returncode == 0:
        return CodeExecutionResult(passed=True, reason="ok")
    stderr_tail = stderr.decode("utf-8", errors="replace").strip().splitlines()
    detail = stderr_tail[-1] if stderr_tail else ""
    return CodeExecutionResult(
        passed=False, reason=f"exit code {process.returncode}: {detail}"
    )


def score_mbpp_response(
    response_text: str,
    test_list: Sequence[str],
    test_imports: Sequence[str] = (),
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
) -> bool:
    """Score an MBPP response: extract the code, run the tests, return pass/fail.

    The boolean mirrors `src.scoring.score_response`'s interface so the eval
    scripts and the W&B logging schema need no structural change (checklist
    2.2). Use `run_python_with_tests` directly when the failure `reason` is
    needed.
    """
    code = extract_python_code(response_text)
    if code is None:
        return False
    return run_python_with_tests(
        code=code,
        test_list=test_list,
        test_imports=test_imports,
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
    ).passed
