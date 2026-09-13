from __future__ import annotations

import sys

from sandbox.cli import main as cli_main


def entrypoint() -> int:
    # No arguments opens the interactive console.
    # Existing CLI arguments retain their original behavior.
    if len(sys.argv) == 1:
        from sandbox.console import run_console
        return run_console()
    return cli_main()


raise SystemExit(entrypoint())
