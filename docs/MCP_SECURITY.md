# MCP security — first hardening pass

Install `python -m pip install -e .`, then start with
`python -m sandbox.mcp_server`. The root `mcp_server.py` is only a compatibility
launcher for that same implementation. MCP SDK 2.2.0 is pinned in project dependencies.
The server binds only to 127.0.0.1:8001 using Streamable HTTP. It has no authentication;
do not expose it through a public tunnel or reverse proxy. No tunnel is configured.

## Allowed READ operations

- `sandbox_status()`: logical repository identifier and read-only mode.
- `list_market_state_results()`: existing names from a fixed server-owned list:
  `gmm_v1.json`, `state_interpretation_v1.json`, `future_behavior_v1.json`,
  `future_behavior_validation_v1.json`, `state_transition_discovery_v1.json`.

The listing publishes names only, never file contents. It does not discover new
publications automatically. Adding a name requires code review. Unknown files,
H5 broker reports, backups, databases, raw data, 2026 diagnostics, and sealed
artifacts are excluded. Symlinked publications and redirected result directories
are excluded. Filesystem errors return an empty list without local path details.
Neither tool accepts any arguments, paths or partition IDs.

## Forbidden

Arbitrary filesystem reads/writes, Python/shell/SQL execution, Git operations,
model fitting/training, strategy or dataset modification, protected partition or
Final Holdout access, MT5/live trading, and security-policy modification are not
MCP capabilities. No execution tools are registered.

Current MCP is NOT approved for controlled research execution. A future execution
interface requires authenticated authorization, Guardian/preflight checks before
any protected read or plugin evaluation, and an isolated worker with no Final
Holdout filesystem access and no write access to frozen artifacts. Current Python
application checks do not contain malicious local code. Read-only service methods
must also be checked for database initialization/audit-write side effects.

Sealed executions now complete authorization, frozen bindings, contamination,
strategy fingerprints and persistent repeat claims before loading data/evaluating
plugins. Failed attempts retain claims and require separate human review; no
automatic retry unlock is provided. The CLI path check denies access when protection
metadata cannot be established. Only a genuinely new catalog is initialized through
the existing service schema; an existing malformed catalog is not silently repaired.
