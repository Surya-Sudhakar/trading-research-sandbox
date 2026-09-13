from __future__ import annotations

from sandbox import console


def scripted(*answers: str):
    it = iter(answers)
    return lambda _prompt="": next(it)


def test_console_exit_cleanly():
    assert console.run_console(input_fn=scripted("0")) == 0


def test_invalid_selection_then_exit(capsys):
    assert console.run_console(input_fn=scripted("x", "0")) == 0
    assert "Invalid selection." in capsys.readouterr().out


def test_strategies_list_routes_to_existing_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(console, "cli_main", lambda argv: calls.append(argv) or 0)
    assert console.run_console(input_fn=scripted("4", "1", "", "0", "0")) == 0
    assert calls == [["strategies", "list"]]


def test_validation_run_routes_to_sealed_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(console, "cli_main", lambda argv: calls.append(argv) or 0)
    answers = scripted("7", "2", "CAND-1", "PART-VAL-1", "validation-intents.json", "VALIDATE", "", "0", "0")
    assert console.run_console(input_fn=answers) == 0
    assert calls == [["research", "validation", "run", "CAND-1", "PART-VAL-1", "validation-intents.json"]]


def test_final_run_requires_confirmation(monkeypatch):
    calls = []
    monkeypatch.setattr(console, "cli_main", lambda argv: calls.append(argv) or 0)
    answers = scripted("8", "2", "CAND-1", "PART-FINAL-1", "final-intents.json", "NO", "", "0", "0")
    assert console.run_console(input_fn=answers) == 0
    assert calls == []


def test_eof_exits_cleanly():
    def eof(_prompt=""):
        raise EOFError
    assert console.run_console(input_fn=eof) == 0
