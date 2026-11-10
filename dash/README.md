# dash — always-up monitoring for the basalt serve wrapper

The basalt web UI normally lives inside `python -m serve.server`, whose lifetime equals one
engine session: the host flip tears the wrapper down, and with it the port the dashboard is on.
`dash/` is a standalone observer sidecar (same pattern as `dash/` in the capy-flash repo for
nInfer) that keeps the monitoring UI answering **for the whole life of the container**, through
flips, pauses, wrapper restarts and engine-down windows.

## Contract

| route | behaviour when the wrapper is up | behaviour when it is down |
|---|---|---|
| `GET /`, `/web/*`, `/fonts/*` | static UI from `--web` | same (always up) |
| `GET /metrics`, `GET /slots` | relayed verbatim from `--wrapper`, incl. `Authorization` passthrough | `503 {"error":{"message":"wrapper unavailable ..."}}` — the UI renders its Offline pill |
| `GET /api/logs?after=&limit=&level=` | **this dash's own ring**, fed by tailing `--log`: one id stream across every wrapper session | same (always up, keeps growing) |
| `POST /v1/load` | relayed verbatim (the UI's hot-load control; the dash adds no powers) | `503` |
| any other `POST` | `405` | `405` |
| `GET /health` | this dash: `{service:"basalt-dash", engine_up, api_key, model}` | same |

Read-only observer apart from relaying the UI's own `/v1/load`: it never spawns, signals, restarts
or reads secrets from the engine or wrapper. The wrapper's `api_key` and `model` come from its
public `/health`.

## Files

- `server.py` — the sidecar (stdlib only; no psutil: hardware facts arrive inside the relayed
  `/metrics` itself, which is why this dash needs no GPU/host telemetry of its own).
- `loglevel.py`, `logbuf.py` — vendored verbatim from `serve/util/` (only the intra-package
  import rewritten), so level classification (`basalt area: warning:` / `[server] warning:` /
  tracebacks), the ring, ANSI stripping and `tail_file` are byte-identical to the wrapper's own
  semantics.
- `tests/test_dash.py` — relay contract, offline gating, ring/level/filter semantics, static +
  font-route guards, `/v1/load` passthrough, 405 policy.

    cd dash && python3 -m unittest discover -s tests -p 'test_*.py'

## Run

    python3 dash/server.py --listen 0.0.0.0:8099 --wrapper 127.0.0.1:8097 \
        --log /var/log/basalt-serve.log --web /root/basalt/serve/web

`--web` points at the checkout's `serve/web` on purpose: the displayed UI then always matches
the engine build running on the box, and no second copy of the web assets can drift.

Notes:
- The wrapper log carries no timestamps; ring timestamps are arrival times — the wrapper stamps
  its own panel the same way, and `/api/logs` ids are monotonic per dash run (the UI's restart
  detection works off `last` going backwards).
- Provenance: this directory is the basalt-side sibling of the `dash/` observer in
  `azido-azide-azide/capy-flash`. Both serve the same `/metrics` + `/api/logs` UI contract; the
  nInfer one parses a text log because nInfer exposes no metrics endpoint, this one relays
  because basalt does.
