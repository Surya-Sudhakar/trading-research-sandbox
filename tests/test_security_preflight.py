from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sandbox.catalog import Catalog
from sandbox.cli import _assert_unprotected_market_path
from sandbox.research.errors import ResearchError
from sandbox.strategies.examples.synthetic_test import SyntheticEveryNthBar
from sandbox.strategies.registry import StrategyRegistry
from sandbox.strategies.models import WarmupRequirements
from test_stage6_partitions import source_fixture,freeze_for_validation,validated_fixture,validation_result_id


@pytest.mark.parametrize("kind",["unavailable","missing_schema","malformed","dangling_source","unknown_role"])
def test_path_check_fails_closed(tmp_path,kind):
    catalog=Catalog(tmp_path/"catalog.db")
    if kind=="unavailable":
        catalog.connection=Mock(side_effect=OSError("private path"))
    elif kind in {"malformed","dangling_source","unknown_role"}:
        catalog.initialize()
        with catalog.connection() as con:
            con.execute("CREATE TABLE data_partitions(path,source_dataset_id,role)")
            con.execute("INSERT INTO data_partitions VALUES (?,?,?)",("" if kind=="malformed" else "protected.parquet","missing","UNKNOWN" if kind=="unknown_role" else "VALIDATION"))
    with pytest.raises(ResearchError,match="CHECK_FAILED") as error:
        _assert_unprotected_market_path(catalog,tmp_path/"input.parquet")
    assert "private path" not in str(error.value)


def test_known_empty_protection_catalog_allows_normal_path(tmp_path):
    catalog=Catalog(tmp_path/"catalog.db");catalog.initialize()
    with catalog.connection() as con:con.execute("CREATE TABLE data_partitions(path,source_dataset_id,role)")
    _assert_unprotected_market_path(catalog,tmp_path/"input.parquet")


class CounterPlugin(SyntheticEveryNthBar):
    metadata=SyntheticEveryNthBar.metadata.model_copy(update={"supported_symbols":("X",),"warmup_requirements":WarmupRequirements(bars_by_timeframe={"M1":1})})
    calls=0
    def evaluate(self,context):
        self.calls+=1
        raise RuntimeError("EVALUATION_REACHED")


def setup(role,plugin_type=CounterPlugin):
    if role=="validation":
        reg,control,service,prov,path,raw,d,v,f,c,e=source_fixture()
    else:
        reg,control,service,prov,d,v,f,c,e=validated_fixture()
    plugin=plugin_type();registry=StrategyRegistry();registry.register(plugin)
    _,code,params=registry.fingerprint(plugin,plugin.metadata.default_parameters)
    spec={"strategy_id":plugin.metadata.strategy_id,"strategy_version":plugin.metadata.strategy_version,
          "parameters":{},"strategy_code_fingerprint":code,"parameter_fingerprint":params}
    data=c.model_dump(mode="json");data["strategy_spec"]=spec
    with reg.catalog.connection() as con:con.execute("UPDATE candidates SET record_json=? WHERE candidate_id=?",(json.dumps(data),c.candidate_id))
    c=control.get_candidate(c.candidate_id)
    if role=="validation":
        c=freeze_for_validation(control,c,prov);service.bind_validation(c.candidate_id,v.partition_id);partition=v
    else:
        service.authorize_final(c.candidate_id,f.partition_id,validation_result_id(reg,c.candidate_id),"test");partition=f
    return reg,service,c,partition,plugin,registry


@pytest.mark.parametrize("role",["validation","final"])
@pytest.mark.parametrize("reason",["config","evaluation","strategy","contamination","claim","role","checksum","authorization"])
def test_rejected_preflight_never_reads_or_evaluates(role,reason,monkeypatch):
    reg,service,c,p,plugin,registry=setup(role)
    if reason in {"config","evaluation","strategy"}:
        data=c.model_dump(mode="json")
        if reason=="strategy":data["strategy_spec"]["strategy_code_fingerprint"]="changed"
        else:data["execution_config" if reason=="config" else "evaluation_spec"]={"changed":True}
        with reg.catalog.connection() as con:con.execute("UPDATE candidates SET record_json=? WHERE candidate_id=?",(json.dumps(data),c.candidate_id))
    elif reason=="contamination":monkeypatch.setattr(service,"_lineage_contaminated",lambda *_:True)
    elif reason=="claim":
        with reg.catalog.connection() as con:con.execute(f"INSERT INTO {role}_execution_claims VALUES (?,?,?,?)",(c.candidate_id,p.partition_id,"RUNNING","test"))
    elif reason=="authorization":
        with reg.catalog.connection() as con:
            con.execute("DELETE FROM "+("partition_bindings" if role=="validation" else "final_authorizations"))
    elif reason=="role":
        original=service._get_partition(p.partition_id)
        monkeypatch.setattr(service,"_get_partition",lambda _:original.model_copy(update={"role":"DISCOVERY"}))
    elif reason=="checksum":monkeypatch.setattr("sandbox.partition.service.sha256_file",lambda _:"bad")
    read=Mock(side_effect=AssertionError("protected read before authorization"))
    monkeypatch.setattr("sandbox.partition.service.pd.read_parquet",read)
    with pytest.raises(ResearchError):getattr(service,f"sealed_{role}_strategy_execute")(c.candidate_id,p.partition_id,registry)
    assert plugin.calls==0
    read.assert_not_called()


@pytest.mark.parametrize("role",["validation","final"])
def test_claim_exists_before_plugin_and_failed_plugin_cannot_repeat(role):
    reg,service,c,p,plugin,registry=setup(role)
    execute=getattr(service,f"sealed_{role}_strategy_execute")
    with pytest.raises(RuntimeError,match="EVALUATION_REACHED"):execute(c.candidate_id,p.partition_id,registry)
    assert plugin.calls==1
    with reg.catalog.connection() as con:
        assert con.execute(f"SELECT status FROM {role}_execution_claims WHERE candidate_id=? AND partition_id=?",(c.candidate_id,p.partition_id)).fetchone()[0]=="RUNNING"
    with pytest.raises(ResearchError,match="already claimed"):execute(c.candidate_id,p.partition_id,registry)
    assert plugin.calls==1


class EmittingPlugin(CounterPlugin):
    def evaluate(self,context):
        from sandbox.strategies.models import StrategySignal
        from sandbox.execution.models import Direction,EntryType
        self.calls+=1
        bar=context.current_closed_bar("X","M1");price=float(bar.values["close"])
        return (StrategySignal(signal_id="test-"+context.current_time.isoformat(),setup_id="test",
            symbol="X",decision_timestamp=context.current_time,information_cutoff=context.information_cutoff,
            direction=Direction.LONG,entry_type=EntryType.MARKET_CLOSE,reference_price=price,
            stop_price=price-1,target_price=price+1),)


@pytest.mark.parametrize("role",["validation","final"])
def test_authorized_strategy_completes_with_one_claim_and_repeat_is_denied(role):
    reg,service,c,p,plugin,registry=setup(role,EmittingPlugin)
    execute=getattr(service,f"sealed_{role}_strategy_execute")
    result=execute(c.candidate_id,p.partition_id,registry)
    assert "result" in result and plugin.calls>0
    previous=plugin.calls
    with reg.catalog.connection() as con:
        rows=con.execute(f"SELECT status FROM {role}_execution_claims WHERE candidate_id=? AND partition_id=?",(c.candidate_id,p.partition_id)).fetchall()
        assert [row[0] for row in rows]==["COMPLETED"]
    with pytest.raises(ResearchError):execute(c.candidate_id,p.partition_id,registry)
    assert plugin.calls==previous
