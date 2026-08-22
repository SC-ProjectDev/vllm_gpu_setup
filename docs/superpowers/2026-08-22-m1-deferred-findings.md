# M1 deferred findings (from review ledger, 2026-08-22)

Triage by the final whole-branch review: none of these affect the M1 acceptance run. Revisit in M2.

- Task 1: minor (deferred): _scalar quoted-value branch does not strip trailing inline comment (lib/profile.py ~57-64)
- Task 1: minor (deferred): list-item comment stripping only matches two-space "  #" (lib/profile.py ~85)
- Task 2: minor (deferred): no dedicated test that run_bash leaves caller env unmutated / passes non-path values through (Task 3+ exercise the non-path branch indirectly)
- Task 2: minor (deferred): conftest BASH=shutil.which("bash") gives opaque TypeError if bash missing
- Task 3: minor (deferred): serve.sh has no trap to kill $vllm_pid if serve.sh itself is terminated
- Task 3: minor (deferred): mapfile from profile.py returns 0 on a failed python run; a malformed profile could launch vllm with partial argv
- Task 4: minor (deferred): .env inline trailing comments (`KEY=value # note`) become part of the value
- Task 5: minor (deferred): curl failure exits with curl's code (7/22/28) not 1; no test for non-200 path
- Task 5: minor (deferred): $MODEL interpolated unescaped into the JSON heredoc
- Task 6: minor (deferred): pid_alive Windows check is a substring match on tasklist output (locale-fragile); CSV parse would be sturdier
- Task 6: minor (deferred): ssh_run returns stdout-or-stderr and drops the return code (Tasks 7/8 callers must tolerate this)
- Task 6: minor (deferred): unused `import json` in tests/test_gpu_llm.py
- Task 7: minor (deferred): ssh_run inside wait_ready uses a fixed 20s timeout independent of interval
- Task 7: minor (deferred): no test of a STARTING→FAILED transition mid-poll (fake ssh output is static)
- Task 5: minor (deferred): implementer of Task 7 reports 3 pre-existing ResourceWarnings in tests/test_smoke.py (HTTPServer socket not server_close()'d) — final review to triage
- Task 7: minor (deferred): _spawned list never pruned after down (harmless per-process)
- Task 7: minor (deferred): test server cleanup relies on manual _stop() calls rather than a yield fixture
- Task 8: minor (deferred): `models: (none)` is a third output form not in the spec table
- Task 8: minor (deferred): no test for `models: unreachable` branch
- Task 8: minor (deferred): cmd_status prints ssh_run's full multi-line output for gpu: rather than the first line
- Ruling: minors 10-16 and the rest of the deferred list stay deferred to M2 — why: none affect the acceptance run — cost if wrong: small polish later.
- Final: minor (deferred): "cmd.exe" allowance in pid_alive is a test-shaped hole; replace .bat fake with an exe/py shim in M2
- Final: minor (deferred): .env comment strip eats a quoted value containing " #"
- Final: minor (deferred): detect_gpu's profile table no longer reaches bootstrap.log (only first stderr line surfaced)
- Final: minor (deferred): unwritable .detect.err turns a good detection into a bogus unknown-gpu failure
- Final: minor (deferred): ssh.log opened "w" truncates the previous attempt's stderr on retry
- Final: minor (deferred): immediate-exit tunnel path leaves stale state.json; .env tests write to the real repo root (not xdist-safe)
- Final: deferred to M2 (reviewer minors 10-16): smoke.sh ignores CURL_BIN; serve.sh FAILED-on-exit-0 mismatch; cmd_logs uses config not state; logs tails only vllm.log; Vast-host-can-read-env note; extra_args whitespace split; README wording
