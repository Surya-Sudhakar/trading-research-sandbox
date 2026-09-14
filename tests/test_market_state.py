from __future__ import annotations
from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path
from uuid import uuid4
import math
import pandas as pd
import pytest

from sandbox.market_state import (FEATURE_SCHEMA_VERSION,FeatureConfiguration,FeatureRegistry,
    MarketStateEngine,SnapshotLink,SnapshotPhase,create_snapshot_ledger,snapshot_signals,store_snapshot_ledger)
from sandbox.market_state.swings import confirmed_swings
from sandbox.partition.service import PartitionService
from sandbox.research.errors import ResearchError
from sandbox.strategies.registry import StrategyRegistry
from sandbox.strategies.runtime import StrategyRuntime
from sandbox.strategies.s001 import BASELINE_PARAMETERS,S001Strategy


def candles(n=320,freq="15min",start="2024-01-01",flat=False):
    index=pd.date_range(start,periods=n,freq=freq,tz="UTC");base=[100 if flat else 100+i*.1+((i%5)-2)*.02 for i in range(n)]
    return pd.DataFrame({"timestamp_utc":index,"open":base,"high":[x+(0 if flat else .3) for x in base],
        "low":[x-(0 if flat else .2) for x in base],"close":[x+(0 if flat else .1) for x in base],
        "spread":[0]*n,"tick_volume":[1]*n,"real_volume":[0]*n})


def aligned_frames(n=320):
    m15=candles(n);h1=PartitionService._strategy_timeframe_frames(m15,S001Strategy(),"M15")["H1"]
    return {"M15":m15,"H1":h1}


def snap(frames=None,cutoff=None,config=None,**kw):
    frames=frames or aligned_frames();cutoff=cutoff or (frames["M15"].timestamp_utc.iloc[-1]+timedelta(minutes=15)).to_pydatetime()
    return MarketStateEngine(config).snapshot(frames,symbol="EURUSD",decision_timestamp=cutoff,information_cutoff_timestamp=cutoff,**kw)


def test_framework_is_independent_registry_valid_and_schema_explicit():
    registry=FeatureRegistry(FeatureConfiguration());assert registry.validate() and registry.definitions() and FEATURE_SCHEMA_VERSION=="MARKET_STATE_V1"
    assert "sandbox.strategies.s001" not in MarketStateEngine.__module__
    snapshot=snap();registered={x.feature_name for x in registry.definitions()};assert set(snapshot.values)<=registered and set(snapshot.validity_flags)<=registered


def test_configuration_and_snapshot_identity_deterministic_and_material_changes():
    a=MarketStateEngine();b=MarketStateEngine();assert a.configuration_fingerprint==b.configuration_fingerprint and a.code_fingerprint==b.code_fingerprint
    changed=MarketStateEngine(FeatureConfiguration(horizons=(4,8)));assert changed.configuration_fingerprint!=a.configuration_fingerprint
    one=snap();two=snap();assert one.snapshot_id==two.snapshot_id and one.values==two.values
    later=snap(cutoff=(aligned_frames()["M15"].timestamp_utc.iloc[-2]+timedelta(minutes=15)).to_pydatetime());assert later.snapshot_id!=one.snapshot_id


def test_snapshot_and_configuration_are_immutable_and_outcome_is_rejected():
    snapshot=snap()
    with pytest.raises(TypeError):snapshot.values["x"]=1
    with pytest.raises(FrozenInstanceError):snapshot.symbol="X"
    with pytest.raises(AttributeError):snapshot.__dict__
    config=FeatureConfiguration()
    with pytest.raises(FrozenInstanceError):config.atr_period=2


def test_future_and_unfinished_bars_are_excluded_for_m15_and_h1():
    frames=aligned_frames(40);cutoff=pd.Timestamp("2024-01-01 05:45",tz="UTC").to_pydatetime();snapshot=snap(frames,cutoff)
    assert any("M15:UNFINISHED" in x for x in snapshot.data_quality_flags) and any("H1:UNFINISHED" in x for x in snapshot.data_quality_flags)
    future=frames["M15"].copy();future.loc[len(future)]={**future.iloc[-1].to_dict(),"timestamp_utc":pd.Timestamp("2030-01-01",tz="UTC"),"close":999999}
    assert snap({**frames,"M15":future},cutoff).values==snapshot.values


def test_information_cutoff_cannot_follow_decision_and_protected_phase_denied():
    now=pd.Timestamp("2024-01-02",tz="UTC").to_pydatetime()
    with pytest.raises(ResearchError,match="FUTURE_CUTOFF"):MarketStateEngine().snapshot(aligned_frames(),symbol="X",decision_timestamp=now,information_cutoff_timestamp=now+timedelta(minutes=1))
    with pytest.raises(ResearchError,match="PROTECTED_MARKET_STATE"):MarketStateEngine().snapshot(aligned_frames(),symbol="X",decision_timestamp=now,research_phase="VALIDATION")


def test_trend_displacement_slope_efficiency_and_atr_normalization_are_manual():
    frame=candles(30);cutoff=(frame.timestamp_utc.iloc[-1]+timedelta(minutes=15)).to_pydatetime();s=snap({"M15":frame},cutoff,FeatureConfiguration(requested_timeframes=("M15",),horizons=(4,),atr_period=3,volatility_short_window=3,volatility_long_window=5,volatility_percentile_history=10))
    closes=frame.close.iloc[-5:].tolist();expected=closes[-1]-closes[0];path=sum(abs(b-a) for a,b in zip(closes,closes[1:]))
    assert s.values["M15.h4.displacement_price"]==pytest.approx(expected)
    assert s.values["M15.h4.directional_efficiency"]==pytest.approx(abs(expected)/path)
    import numpy as np
    assert s.values["M15.h4.regression_slope_price_per_bar"]==pytest.approx(np.polyfit(range(5),closes,1)[0])
    assert s.values["M15.h4.displacement_atr"]==pytest.approx(expected/s.values["M15.volatility.atr"])


def test_zero_efficiency_atr_and_range_are_explicitly_unavailable():
    frame=candles(30,flat=True);s=snap({"M15":frame},(frame.timestamp_utc.iloc[-1]+timedelta(minutes=15)).to_pydatetime(),FeatureConfiguration(requested_timeframes=("M15",),horizons=(4,),atr_period=3,volatility_short_window=3,volatility_long_window=5,volatility_percentile_history=10))
    assert not s.validity_flags["M15.h4.directional_efficiency"] and not s.validity_flags["M15.volatility.atr"] and not s.validity_flags["M15.h4.close_position_in_range"]
    assert "M15.h4.directional_efficiency" not in s.values


def test_volatility_atr_tr_stats_ratio_and_percentile_are_causal():
    frame=candles(80);cfg=FeatureConfiguration(requested_timeframes=("M15",),horizons=(4,),atr_period=5,volatility_short_window=5,volatility_long_window=10,volatility_percentile_history=20)
    cutoff=(frame.timestamp_utc.iloc[50]+timedelta(minutes=15)).to_pydatetime();a=snap({"M15":frame},cutoff,cfg)
    changed=frame.copy();changed.loc[55:,"high"]=9999;b=snap({"M15":changed},cutoff,cfg)
    assert a.values==b.values and a.values["M15.volatility.atr"]>0 and a.values["M15.volatility.tr_mean_short"]>0
    assert 0<a.values["M15.volatility.atr_percentile"]<=100 and a.values["M15.volatility.short_long_atr_ratio"]>0


def test_momentum_recent_preceding_sign_change_and_missing_denominator():
    frame=candles(20,flat=True);frame.loc[:8,"close"]=list(range(100,109));frame.loc[9:,"close"]=list(range(108,97,-1))[:11];frame["high"]=frame[["open","close"]].max(axis=1)+.2;frame["low"]=frame[["open","close"]].min(axis=1)-.2
    cfg=FeatureConfiguration(requested_timeframes=("M15",),horizons=(4,),atr_period=3,volatility_short_window=3,volatility_long_window=5,volatility_percentile_history=5)
    s=snap({"M15":frame},(frame.timestamp_utc.iloc[-1]+timedelta(minutes=15)).to_pydatetime(),cfg)
    assert s.values["M15.h4.momentum_change.recent_displacement_price"]<0
    # A dedicated V-shaped window establishes sign-change behavior without interpreting it.
    v=frame.iloc[:9].copy();v["close"]=[100,101,102,103,104,103,102,101,100];v["high"]=v[["open","close"]].max(axis=1)+.2;v["low"]=v[["open","close"]].min(axis=1)-.2
    sv=snap({"M15":v},(v.timestamp_utc.iloc[-1]+timedelta(minutes=15)).to_pydatetime(),cfg)
    assert sv.values["M15.h4.momentum_change.displacement_sign_changed"] is True


def test_range_extremes_distances_and_maturity_are_manual():
    frame=candles(20);cfg=FeatureConfiguration(requested_timeframes=("M15",),horizons=(4,),atr_period=3,volatility_short_window=3,volatility_long_window=5,volatility_percentile_history=5);s=snap({"M15":frame},(frame.timestamp_utc.iloc[-1]+timedelta(minutes=15)).to_pydatetime(),cfg)
    tail=frame.iloc[-4:];hi=tail.high.max();lo=tail.low.min();close=tail.close.iloc[-1]
    assert s.values["M15.h4.rolling_range_price"]==pytest.approx(hi-lo) and s.values["M15.h4.close_position_in_range"]==pytest.approx((close-lo)/(hi-lo))
    assert s.values["M15.h4.bars_since_rolling_high"]==0 and s.values["M15.h4.distance_from_rolling_low_price"]==pytest.approx(close-lo)


def test_confirmed_pivot_has_exact_right_bar_delay_and_distances():
    highs=[1,2,5,2,1];lows=[0,.5,1,.5,0]
    assert not confirmed_swings(highs[:4],lows[:4],2,2)
    swings=confirmed_swings(highs,lows,2,2);assert swings and swings[0].kind=="HIGH" and swings[0].confirmation_index==4
    frame=candles(12,flat=True);frame["high"]=[1,2,5,2,1,2,3,2,1,2,3,2];frame["low"]=[0,.5,1,.5,0,.5,1,.5,0,.5,1,.5];frame["open"]=1.5;frame["close"]=1.5
    cfg=FeatureConfiguration(requested_timeframes=("M15",),horizons=(4,),atr_period=3,volatility_short_window=3,volatility_long_window=5,volatility_percentile_history=5)
    s=snap({"M15":frame},(frame.timestamp_utc.iloc[-1]+timedelta(minutes=15)).to_pydatetime(),cfg)
    assert s.values["M15.swing.high.price"]==3 and s.values["M15.swing.high.bars_since_confirmation"]==3
    assert s.values["M15.swing.high.distance_price"]==pytest.approx(-1.5) and "M15.structure.lower_high_count" in s.values


def test_insufficient_history_no_forward_fill_and_nonfinite_fails_safely():
    frame=candles(3);cfg=FeatureConfiguration(requested_timeframes=("M15",),horizons=(4,),atr_period=14,volatility_short_window=14,volatility_long_window=50,volatility_percentile_history=250)
    s=snap({"M15":frame},(frame.timestamp_utc.iloc[-1]+timedelta(minutes=15)).to_pydatetime(),cfg);assert not s.validity_flags["M15.h4.displacement_price"] and "M15.h4.displacement_price" not in s.values
    frame.loc[1,"close"]=math.nan
    with pytest.raises(ResearchError,match="NON_FINITE"):snap({"M15":frame},(frame.timestamp_utc.iloc[-1]+timedelta(minutes=15)).to_pydatetime(),cfg)


def test_multitimeframe_alignment_h1_becomes_available_only_at_close():
    m15=candles(8);h1=PartitionService._strategy_timeframe_frames(m15,S001Strategy(),"M15")["H1"];cfg=FeatureConfiguration(requested_timeframes=("M15","H1"),horizons=(1,),atr_period=1,volatility_short_window=1,volatility_long_window=2,volatility_percentile_history=2)
    before=snap({"M15":m15,"H1":h1},pd.Timestamp("2024-01-01 00:45",tz="UTC").to_pydatetime(),cfg);at=snap({"M15":m15,"H1":h1},pd.Timestamp("2024-01-01 01:00",tz="UTC").to_pydatetime(),cfg)
    assert not before.validity_flags["H1.h1.displacement_price"] and not at.validity_flags["H1.h1.displacement_price"]
    after=snap({"M15":m15,"H1":h1},pd.Timestamp("2024-01-01 02:00",tz="UTC").to_pydatetime(),cfg);assert after.validity_flags["H1.h1.displacement_price"]


def test_calendar_new_york_dst_weekday_hour_and_time_block_position():
    frame=candles(20,start="2024-03-10");cfg=FeatureConfiguration(requested_timeframes=("M15",),horizons=(4,),atr_period=3,volatility_short_window=3,volatility_long_window=5,volatility_percentile_history=5)
    a=snap({"M15":frame},pd.Timestamp("2024-03-10 06:30",tz="UTC").to_pydatetime(),cfg);b=snap({"M15":frame},pd.Timestamp("2024-03-10 07:30",tz="UTC").to_pydatetime(),cfg)
    assert a.values["calendar.new_york_hour"]==1 and b.values["calendar.new_york_hour"]==3 and a.values["calendar.day_of_week"]==6
    assert a.values["calendar.block_180m_elapsed_fraction"]==pytest.approx(1/6)


def test_snapshot_ledger_is_immutable_reproducible_and_round_trips_artifacts():
    s=snap();link=SnapshotLink(s.snapshot_id,SnapshotPhase.ENTRY_DECISION,"SETUP","SIGNAL","TRADE");a=create_snapshot_ledger([s],[link]);b=create_snapshot_ledger([s],[link]);assert a.ledger_fingerprint==b.ledger_fingerprint
    root=Path("tests/runtime")/("market-state-"+uuid4().hex);stored=store_snapshot_ledger(a,root,"LEDGER");assert stored.snapshot_path.exists() and len(pd.read_parquet(stored.snapshot_path))==1
    with pytest.raises(FileExistsError):store_snapshot_ledger(a,root,"LEDGER")


def test_s001_external_linkage_does_not_change_signal_or_fingerprints():
    from test_strategy_s001 import market_rows
    h1,m15=market_rows();registry=StrategyRegistry();plugin=S001Strategy();registry.register(plugin);before=registry.fingerprint(plugin,dict(BASELINE_PARAMETERS));run=StrategyRuntime(registry).run("S001",{"H1":h1,"M15":m15},symbol="EURUSD",dataset_id="SYNTH",experiment_id="TEST")
    engine=MarketStateEngine(FeatureConfiguration(horizons=(4,8,12),volatility_long_window=14,volatility_percentile_history=20));snapshots,links=snapshot_signals(engine,{"H1":h1,"M15":m15},run.signal_ledger,symbol="EURUSD",dataset_id="SYNTH")
    after=registry.fingerprint(plugin,dict(BASELINE_PARAMETERS));assert snapshots and links[0].signal_id==run.signal_ledger.records[0].signal_id and before==after
    rerun=StrategyRuntime(registry).run("S001",{"H1":h1,"M15":m15},symbol="EURUSD",dataset_id="SYNTH",experiment_id="TEST");assert run.signal_ledger==rerun.signal_ledger


def test_engine_has_no_filter_ranking_optimizer_ml_outcome_or_stage7_surface():
    engine=MarketStateEngine();snapshot=snap()
    assert not any(x in dir(engine) for x in ("filter","rank_features","optimize","fit","predict","trade","take_trade"))
    assert not any(any(token in k.lower() for token in ("win","loss","pnl","mfe","mae")) for k in snapshot.values)
    with pytest.raises(ResearchError,match="STAGE7"):engine.stage7_results()
    with pytest.raises(ResearchError,match="PARTITION_ACCESS"):engine.request_partition()


def test_visible_bad_ohlc_flagged_but_future_bad_ohlc_cannot_contaminate_cache():
    frame=candles(20);cutoff=(frame.timestamp_utc.iloc[10]+timedelta(minutes=15)).to_pydatetime();cfg=FeatureConfiguration(requested_timeframes=("M15",),horizons=(4,),atr_period=3,volatility_short_window=3,volatility_long_window=5,volatility_percentile_history=5)
    base=snap({"M15":frame},cutoff,cfg);future=frame.copy();future.loc[15,["high","low"]]=[-999,999];assert snap({"M15":future},cutoff,cfg).values==base.values
    visible=frame.copy();visible.loc[5,["high","low"]]=[-999,999]
    assert "M15:INVALID_OHLC_GEOMETRY_OBSERVED" in snap({"M15":visible},cutoff,cfg).data_quality_flags
