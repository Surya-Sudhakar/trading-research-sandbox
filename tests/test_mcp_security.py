import asyncio
import inspect
import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
from sandbox import mcp_server as server


def test_only_two_no_argument_tools():
    tools=asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools}=={"sandbox_status","list_market_state_results"}
    for tool in tools:
        assert not tool.input_schema.get("properties")
        assert not inspect.signature(getattr(server,tool.name)).parameters


def test_status_has_no_local_paths():
    result=server.sandbox_status()
    assert result=={"status":"ok","repository":"trading-research-sandbox","mode":"read-only"}
    text=json.dumps(result)
    assert str(server.ROOT) not in text and "C:" not in text and "Users" not in text


def test_only_explicit_publications_never_read_contents(tmp_path,monkeypatch):
    monkeypatch.setattr(server,"ROOT",tmp_path)
    folder=tmp_path/"results"/"market_state";folder.mkdir(parents=True)
    for name in (*server.PUBLICATIONS,"future_secret_2025.json",".env","model.db","report.xlsx","backup.h5backup","unknown.json"):
        (folder/name).write_text("private content")
    monkeypatch.setattr(Path,"read_text",Mock(side_effect=AssertionError("no contents")))
    monkeypatch.setattr(Path,"read_bytes",Mock(side_effect=AssertionError("no contents")))
    assert server.list_market_state_results()==["results/market_state/"+x for x in server.PUBLICATIONS]


def test_missing_publication_and_filesystem_error_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(server,"ROOT",tmp_path)
    assert server.list_market_state_results()==[]
    monkeypatch.setattr(Path,"resolve",Mock(side_effect=OSError("C:/Users/private/secret")))
    assert server.list_market_state_results()==[]


def test_tools_reject_path_code_partition_arguments():
    for function in (server.sandbox_status,server.list_market_state_results):
        for name in ("path","filename","code","shell","sql","partition_id"):
            with pytest.raises(TypeError):function(**{name:"../../final_vault"})


def test_launcher_has_no_second_server_and_startup_is_local(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    main=Mock();monkeypatch.setattr(server,"main",main)
    namespace=runpy.run_path(str(root/"mcp_server.py"),run_name="compatibility_import")
    assert "mcp" not in namespace
    main.assert_not_called()
    runpy.run_path(str(root/"mcp_server.py"),run_name="__main__")
    main.assert_called_once_with()


def test_canonical_startup(monkeypatch):
    runner=Mock();monkeypatch.setattr(server.mcp,"run",runner)
    server.main()
    runner.assert_called_once_with(transport="streamable-http",host="127.0.0.1",port=8001)


def test_linked_publications_are_not_exposed(tmp_path,monkeypatch):
    monkeypatch.setattr(server,"ROOT",tmp_path)
    folder=tmp_path/"results"/"market_state";folder.mkdir(parents=True)
    (folder/server.PUBLICATIONS[0]).write_text("not published via link")
    monkeypatch.setattr(Path,"is_symlink",lambda self: self.name==server.PUBLICATIONS[0])
    assert server.list_market_state_results()==[]
