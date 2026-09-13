from __future__ import annotations
import json
from pathlib import Path
from uuid import uuid4
from sandbox.external_import import ExternalImportGateway,ImportSpec,Resolution,PriceType
from sandbox.catalog import Catalog
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError
from sandbox.api.app import LOCAL_ORIGINS,create_app
from sandbox.api.models import SystemStatusDTO
from sandbox.api.service import SandboxReadService
from sandbox.config import Settings
from sandbox.strategies.registry import default_registry
from sandbox.strategies.s001 import BASELINE_PARAMETERS

@pytest.fixture(scope="module")
def client():
    # These assertions describe a one-import, empty-research fixture, not the user's evolving catalog.
    root=(Path("tests/runtime")/("api-contract-"+uuid4().hex)).resolve();root.mkdir(parents=True)
    source=root/"quarantined.csv"
    source.write_text("timestamp,open,high,low,close\n2024-01-01T00:00:00Z,1.1,1.0,1.0,1.1\n",encoding="utf-8")
    settings=Settings(None,root/"data",root/"catalog.sqlite3",7,3,"ERROR")
    gateway=ExternalImportGateway(Catalog(settings.catalog_path),settings.data_dir)
    gateway.import_files(ImportSpec((source,),"TEST_PROVIDER","EURUSD","EURUSD",Resolution.M15,
                         PriceType.BID,"UTC",{k:k for k in ("timestamp","open","high","low","close")}))
    return TestClient(create_app(SandboxReadService(settings,probe_broker=False)))

def test_system_status_is_live_safe_read_projection(client):
    value=client.get("/api/system/status");assert value.status_code==200;data=value.json();assert data["source"]=="LIVE_BACKEND_STATE" and data["backend_online"] and data["broker"]["read_only"] and data["strategy_count"]==1

def test_strategy_registry_and_s001_detail_are_authoritative(client):
    rows=client.get("/api/strategies").json();assert [x["strategy_id"] for x in rows]==["S001"]
    s=client.get("/api/strategies/S001").json();assert s["version"]=="1.0.0" and s["required_timeframes"]==["H1","M15"] and s["code_fingerprint"]=="da3cdc50608fcf3bdcb3c1b5e38985d93a30b381ecd549318917caff0b45133a" and s["parameter_fingerprint"]=="2beb1abe68f4583b383dfcd799286c604444885d73a179534685b831abd514cf"

def test_unknown_strategy_error_is_safe(client):
    response=client.get("/api/strategies/../../secrets");assert response.status_code in (404,422) and "Traceback" not in response.text and "Users\\surya" not in response.text

def test_research_state_is_truthfully_empty(client):
    x=client.get("/api/research/status").json();assert (x["hypothesis_count"],x["experiment_count"],x["candidate_count"],x["partition_count"])==(0,0,0,0) and x["discovery_status"]=="NOT_STARTED" and x["final_test_status"]=="SEALED_NOT_ELIGIBLE"

def test_data_status_has_catalog_metadata_without_paths(client):
    response=client.get("/api/data/status");x=response.json();assert x["external_import_gateway_available"] and x["external_production_dataset_count"]==1 and x["imported_datasets"][0]["certification"]=="QUARANTINED" and x["research_partition_count"]==0
    text=response.text.lower();assert "terminal64" not in text and "c:\\" not in text and '"path"' not in text and "password" not in text

def test_empty_evidence_is_not_fabricated(client):
    x=client.get("/api/evidence/status").json();assert not x["available"] and x["report_count"]==0 and x["status"]=="NO_RESEARCH_EVIDENCE_AVAILABLE"

def test_market_state_reports_implemented_but_no_snapshots(client):
    x=client.get("/api/market-state/status").json();assert x["available"] and x["status"]=="IMPLEMENTED_NO_AUTHORIZED_SNAPSHOTS" and x["authorized_snapshot_count"]==0 and x["schema_version"]=="MARKET_STATE_V1"

def test_audit_status_verifies_chains_and_disabled_capabilities(client):
    x=client.get("/api/audit/status").json();assert x["journal_integrity"]==x["access_ledger_integrity"]=="VERIFIED" and x["final_test_vault_state"]=="SEALED" and x["broker_execution"]==x["optimization"]==x["machine_learning"]=="DISABLED"

@pytest.mark.parametrize("path",["/api/final-test/data","/api/validation/candles","/api/partitions/FINAL","/api/files/../../data/catalog.sqlite3","/api/parquet/data","/api/journal","/api/shell","/api/broker/order","/api/strategies/load","/api/market-state/future"])
def test_protected_and_arbitrary_read_routes_do_not_exist(client,path):assert client.get(path).status_code==404

@pytest.mark.parametrize("method,path",[("post","/api/research/journal"),("post","/api/partitions"),("put","/api/strategies/S001"),("post","/api/shell"),("post","/api/broker/order"),("delete","/api/research/experiments/X")])
def test_no_mutation_control_or_execution_routes(client,method,path):assert client.request(method.upper(),path,json={}).status_code in (404,405)

def test_openapi_exposes_only_intended_get_routes(client):
    paths=client.get("/openapi.json").json()["paths"];assert set(paths)=={"/api/system/status","/api/strategies","/api/strategies/{strategy_id}","/api/research/status","/api/data/status","/api/data/import","/api/data/imports/{dataset_id}/diagnostics","/api/evidence/status","/api/audit/status","/api/market-state/status"};assert set(paths["/api/data/import"])=={"post"} and all(set(v)<= {"get","parameters"} for k,v in paths.items() if k!="/api/data/import")

def test_cors_is_restricted_to_local_frontend(client):
    assert "*" not in LOCAL_ORIGINS
    allowed=client.options("/api/system/status",headers={"Origin":"http://127.0.0.1:5173","Access-Control-Request-Method":"GET"});assert allowed.headers.get("access-control-allow-origin")=="http://127.0.0.1:5173"
    denied=client.options("/api/system/status",headers={"Origin":"https://evil.example","Access-Control-Request-Method":"GET"});assert "access-control-allow-origin" not in denied.headers

def test_dtos_reject_extra_sensitive_fields():
    with pytest.raises(ValidationError):SystemStatusDTO.model_validate({"backend_online":True,"source":"LIVE_BACKEND_STATE","broker":{"connected":False,"read_only":True},"research_mode":"X","strategy_count":1,"dataset_count":0,"partition_count":0,"verified_test_count":1,"integrity_status":"VERIFIED","password":"secret"})

def test_safe_internal_error_has_no_diagnostics():
    class Broken(SandboxReadService):
        def __init__(self):pass
        def system(self):raise RuntimeError("C:\\secret\\vault password=hunter2")
    response=TestClient(create_app(Broken()),raise_server_exceptions=False).get("/api/system/status");assert response.status_code==500 and "secret" not in response.text and "hunter2" not in response.text and "Traceback" not in response.text

def test_s001_fingerprints_remain_frozen_after_api_requests(client):
    client.get("/api/strategies/S001");r=default_registry();p=r.resolve("S001");_,code,param=r.fingerprint(p,dict(BASELINE_PARAMETERS));assert code=="da3cdc50608fcf3bdcb3c1b5e38985d93a30b381ecd549318917caff0b45133a" and param=="2beb1abe68f4583b383dfcd799286c604444885d73a179534685b831abd514cf"
