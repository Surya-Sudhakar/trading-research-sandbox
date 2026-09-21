"""Canonical read-only, private-use MCP server. No research services are imported."""
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parent.parent
# Publication names only. No content, paths supplied by callers, or directory discovery.
PUBLICATIONS = (
    "future_behavior_v1.json",
    "future_behavior_validation_v1.json",
    "gmm_v1.json",
    "state_interpretation_v1.json",
    "state_transition_discovery_v1.json",
)
mcp = MCPServer("trading-research-sandbox", version="0.1.0")


@mcp.tool()
def sandbox_status() -> dict:
    """Return logical repository status without local filesystem information."""
    return {"status": "ok", "repository": "trading-research-sandbox", "mode": "read-only"}


@mcp.tool()
def list_market_state_results() -> list[str]:
    """Return names of explicitly published research reports; never their contents."""
    try:
        directory = ROOT / "results" / "market_state"
        if directory.resolve() != directory.absolute():
            return []
        return ["results/market_state/" + name for name in PUBLICATIONS
                if (directory / name).is_file() and not (directory / name).is_symlink()
                and (directory / name).resolve().parent == directory]
    except (OSError, ValueError, RuntimeError):
        # Fail closed without returning OS error text or paths.
        return []


def main() -> None:
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
