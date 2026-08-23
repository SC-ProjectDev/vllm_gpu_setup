# vllm_gpu_setup — Milestone 2 design: `gpu-llm up` / auto-teardown

2026-08-23. Follows the M1 design (`2026-08-22-vllm-gpu-setup-design.md`).
M1 status: accepted on an RTX 5090 2026-08-23 (README table).

## 1. Goal

One command from nothing to a serving model, and one command that
guarantees the meter stops:

- `gpu-llm up` — search Vast offers, confirm, rent, wait for SSH,
  tunnel, wait for READY.
- `gpu-llm down` — kill the tunnel **and destroy the instance** via the
  Vast API.

No FastAPI control plane, no new GPU profiles, no persistent volumes
(all still deferred; vLLM's own API plus ssh covered every M1 need).

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| Vast API access | Raw REST via stdlib `urllib` | Keeps `gpu_llm.py` dependency-free; only 4 endpoints; a break is a small fix. `vastai` CLI/lib rejected as a new dependency with equal plumbing cost |
| Renting | Confirm-first: show best offer, ask y/n; `--yes` skips | `up` spends real money; bad filter match should cost a keypress, not dollars |
| GPU selection | `--gpu 5090\|a100-80\|h100-80\|h200`, default `5090` | Profiles + bootstrap auto-detect already handle all four; only an offer-filter table is new |
| API key | `api_key` in `~/.gpu-llm/config.toml`; `VAST_API_KEY` env overrides | Config file already exists; key never lives in the repo |
| Teardown | `down` destroys by default; `--keep` preserves the instance | The failure mode to design against is a forgotten meter, not an accidental destroy |
| Onstart script | New `vast/onstart.sh`, read by `up` at rent time; `vast/template.md` points at it | Single source — manual and API rentals cannot drift |
| Manual rentals | Still work: `tunnel`/`down` unchanged when state has no `instance_id` | M1 flow remains the fallback when the API path breaks |

## 3. Vast API surface (verified against docs.vast.ai 2026-08-23)

All requests: `Authorization: Bearer <api_key>`, JSON bodies.

| Operation | Request | Notes |
|---|---|---|
| Search offers | `POST https://console.vast.ai/api/v0/bundles` | Body: filter object (below). Response: `{"offers": [...]}` |
| Rent | `PUT https://console.vast.ai/api/v0/asks/{offer_id}` | Body: `image`, `disk`, `env`, `onstart`, `runtype: "ssh"`. Response: `{"new_contract": <instance_id>}` |
| Show instances | `GET https://console.vast.ai/api/v1/instances` | Each instance: `id`, `ssh_host`, `ssh_port`, `actual_status`, `intended_status` |
| Destroy | `DELETE https://console.vast.ai/api/v0/instances/{id}` | Response: `{"success": true, ...}` |

Search filter body (operators `eq/gte/lte/in`; sorted cheapest first):

```json
{
  "limit": 20,
  "type": "ondemand",
  "rentable": {"eq": true},
  "verified": {"eq": true},
  "num_gpus": {"eq": 1},
  "gpu_name": {"in": ["RTX 5090"]},
  "reliability": {"gte": 0.98},
  "inet_down": {"gte": 500},
  "cuda_max_good": {"gte": 12.8},
  "dph_total": {"lte": 1.0},
  "order": [["dph_total", "asc"]]
}
```

`--gpu` maps to `gpu_name` values and per-GPU default price caps:

| `--gpu` | `gpu_name` filter | default `max_price` ($/hr) |
|---|---|---|
| `5090` | `RTX 5090` | 1.00 |
| `a100-80` | `A100 SXM4`, `A100 PCIE` | 1.60 |
| `h100-80` | `H100 SXM`, `H100 PCIE`, `H100 NVL` | 2.50 |
| `h200` | `H200`, `H200 NVL` | 3.50 |

Caps are overridable per-invocation (`--max-price`) or in config
(`[max_price]` table keyed by gpu id). The exact `gpu_name` strings are
implementation-verified against live search results in the first task
(the table above is the design intent; Vast's canonical spellings win).

## 4. CLI surface

```
gpu-llm up   [--gpu ID] [--max-price N] [--yes] [--timeout S] [--interval S]
gpu-llm down [--keep]
tunnel | status | logs   — unchanged from M1
```

- `up` with an existing live tunnel or recorded instance: refuse with
  "already up; run `down` first" (exit 1).
- `down --keep`: tunnel killed, instance left running, instance id kept
  in state so a later plain `down` can still destroy it.
- `status` additionally prints `instance: <id> ($/hr)` when state has one.

## 5. Config & state

`~/.gpu-llm/config.toml` gains:

```toml
api_key = "..."          # or VAST_API_KEY env (env wins)
# [max_price]            # optional per-gpu overrides
# 5090 = 0.80
```

`state.json` (written at rent time, extended by tunnel) gains:

```json
{"instance_id": 12345678, "gpu": "5090", "dph": 0.592,
 "pid": ..., "host": ..., "ssh_port": ..., "local_port": ..., "image": ...}
```

- `up` writes `instance_id`/`gpu`/`dph` immediately after a successful
  rent, **before** waiting for SSH — a crash after renting must leave
  enough state for `down` to destroy.
- Manual-rental state (from bare `tunnel`) simply has no `instance_id`;
  `down` then behaves exactly as in M1 (kills tunnel, prints the
  destroy-it-yourself reminder).
- Stale-state cleanup (M1 deferred finding): `up` and `tunnel` clear
  state.json when its pid is dead and its instance id is absent from
  the live instances list.

## 6. `up` flow

1. Load config; resolve api key (env > config). Missing → exit 1 with
   the console key-page URL.
2. Build the filter for `--gpu`; `POST /bundles`.
3. No offers under the cap → print the three cheapest above it
   (`$/hr`, GPU, Mbps, reliability) and exit 1.
4. Show the best offer: `RTX 5090 · $0.59/hr · 812 Mbps · 99.2% · <geo>`.
   Prompt `rent? [y/N]` unless `--yes`. Decline → exit 0, nothing rented.
5. `PUT /asks/{offer_id}` with: image `vllm/vllm-openai:v0.27.1`,
   `disk: 60`, `runtype: "ssh"`, `env: ""`, `onstart` = contents of
   `vast/onstart.sh`. Record `new_contract` in state.json immediately.
6. Poll `GET /api/v1/instances` (every 10 s, default timeout 10 min)
   until our instance shows `actual_status == "running"` with
   `ssh_host`/`ssh_port` present. Terminal states `exited`, `offline`,
   `unknown` → print status + instance id, ask whether to destroy
   (always interactive, even under `--yes` — a dead-on-arrival
   instance is rare and worth eyes), exit 1.
7. Write `host`/`ssh_port` into config-resolution scope (state, not
   config.toml) and run the existing tunnel + `wait_ready` flow
   unchanged. On READY, print the M1 epilogue plus
   `instance <id> at $<dph>/hr — remember: gpu-llm down destroys it`.
8. Tunnel/READY failure after a successful rent → tunnel is torn down
   (M1 behavior), instance is **not** auto-destroyed; exit message
   names the instance id, the $/hr, and both remedies (`gpu-llm down`
   to destroy, `gpu-llm tunnel` to retry).

## 7. `down` flow

1. M1 behavior: kill tunnel pid if alive.
2. If state has `instance_id` and not `--keep`:
   `DELETE /instances/{id}` → on success print
   `instance <id> destroyed.` and clear state.
3. Destroy API failure → **loud** warning: instance may still be
   billing, id + console URL printed, state kept so `down` can be
   re-run.
4. No `instance_id` in state → M1's current reminder text (manual
   rental: destroy in the console).

## 8. Error handling summary

| Failure | Behavior |
|---|---|
| No/invalid api key | Exit 1, point at https://cloud.vast.ai/manage-keys/ |
| No offers under cap | Exit 1, show 3 cheapest offers above cap |
| Rent request fails (4xx/5xx) | Exit 1, show response body; nothing to clean up |
| Instance never reaches `running` | Ask to destroy, exit 1; state keeps the id either way |
| READY timeout after rent | Tunnel down, instance kept, id + cost + remedies printed |
| Destroy fails | Warning + console link, state kept for retry |
| API/network error mid-poll | Retry within the poll budget; only fail at timeout |

## 9. Testing

Mirror the M1 suite (pytest, no GPU, no live API, offline):

- Pure functions unit-tested directly: `build_offer_query(gpu, cap)`,
  `pick_offer(offers)`, offer formatting, state merge/staleness logic.
- HTTP layer against a local fake Vast server (stdlib `HTTPServer`,
  same pattern as the existing fake status server): search happy path,
  empty offers, rent success (`new_contract`), rent 4xx, poll
  running/exited/timeout, destroy success/failure.
- `cmd_up` confirmation prompt: y / n / `--yes` paths with a scripted
  stdin.
- `cmd_down`: destroy called only when `instance_id` present and no
  `--keep`; state cleared only on success.
- Existing M1 tests must pass unchanged (manual-rental compatibility).

## 10. Acceptance run (paid, ~30 min)

- [ ] `gpu-llm up` end-to-end on a 5090: offer shown, confirmed, READY,
      `llm-cli health`/`chat` work through the tunnel.
- [ ] `gpu-llm status` shows instance id and $/hr.
- [ ] `gpu-llm down` destroys the instance — verify gone in the console
      and by `GET /instances`.
- [ ] `gpu-llm up --yes --max-price 0.30` (absurdly low cap) exits 1
      with the over-cap listing, rents nothing.
- [ ] Record: time from `up` to READY, total cost.

## 11. Out of scope (still deferred)

FastAPI control plane, 3090/4090 profiles (patched vLLM), persistent
weight volumes, idle auto-teardown timers, Ollama/SGLang backends, and
the M1 minor-findings ledger except items in files this milestone
touches (stale state.json is in scope, per §5).
