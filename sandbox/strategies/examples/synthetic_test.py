from sandbox.execution.models import Direction,EntryType
from sandbox.strategies.models import ParameterDefinition,ParameterType,StrategyMetadata,StrategySignal,WarmupRequirements
class SyntheticEveryNthBar:
    metadata=StrategyMetadata(strategy_id="SYNTHETIC_INTERFACE_TEST",strategy_name="Synthetic interface fixture",strategy_version="1.0.0",strategy_family="TEST_ONLY",description="Emits a mechanically predefined signal; not a trading idea.",supported_symbols=("SYNTH",),required_base_timeframe="M1",required_timeframes=("M1",),required_data_fields=("open","high","low","close","spread"),warmup_requirements=WarmupRequirements(bars_by_timeframe={"M1":2}),supported_directions=(Direction.LONG,Direction.SHORT),parameter_schema=(ParameterDefinition(name="every_n",type=ParameterType.INTEGER),),default_parameters={"every_n":2},signal_schema="1",plugin_version="1",capabilities=("LONG","SHORT","MARKET_ENTRY"),analysis_timezone="UTC",synthetic_test_plugin=True)
    def reset(self):self._seen=0
    def evaluate(self,context):
        self._seen+=1
        if self._seen%context.registered_parameters["every_n"]:return ()
        bar=context.current_closed_bar("SYNTH","M1");price=float(bar.values["close"]);stamp=context.current_time
        return (StrategySignal(signal_id=f"SIG-{stamp.isoformat()}",setup_id=f"SETUP-{stamp.isoformat()}",symbol="SYNTH",decision_timestamp=stamp,information_cutoff=context.information_cutoff,direction=Direction.LONG,entry_type=EntryType.MARKET_NEXT_OPEN,reference_price=price,stop_price=price-1,target_price=price+2),)
