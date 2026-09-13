from __future__ import annotations
import inspect
from pathlib import Path
from zoneinfo import ZoneInfo
from sandbox.execution.models import Direction
from sandbox.provenance import sha256_file
from sandbox.research.canonical import sha256_canonical
from sandbox.research.errors import ResearchError
from .models import ParameterType,StrategyMetadata

class StrategyRegistry:
    def __init__(self):self._plugins={}
    def validate(self,plugin):
        if not isinstance(getattr(plugin,"metadata",None),StrategyMetadata):raise ResearchError("INVALID_STRATEGY_PLUGIN: metadata contract missing")
        m=plugin.metadata
        if not callable(getattr(plugin,"reset",None)) or not callable(getattr(plugin,"evaluate",None)):raise ResearchError("INVALID_STRATEGY_PLUGIN: reset/evaluate missing")
        if m.required_base_timeframe not in m.required_timeframes:raise ResearchError("INVALID_STRATEGY_PLUGIN: base timeframe not declared")
        if not m.required_data_fields or not m.supported_directions:raise ResearchError("INVALID_STRATEGY_PLUGIN: data fields/directions required")
        try:ZoneInfo(m.analysis_timezone)
        except Exception as exc:raise ResearchError("INVALID_STRATEGY_PLUGIN: analysis timezone") from exc
        names=[x.name for x in m.parameter_schema]
        if len(names)!=len(set(names)):raise ResearchError("INVALID_STRATEGY_PLUGIN: duplicate parameter")
        self.validate_parameters(plugin,m.default_parameters,allow_missing=True);return True
    def validate_parameters(self,plugin,parameters,allow_missing=False):
        schema={x.name:x for x in plugin.metadata.parameter_schema};unknown=set(parameters)-set(schema)
        if unknown:raise ResearchError("UNKNOWN_STRATEGY_PARAMETERS: "+",".join(sorted(unknown)))
        for name,d in schema.items():
            if name not in parameters:
                if d.required and d.default is None and not allow_missing:raise ResearchError("MISSING_STRATEGY_PARAMETER: "+name)
                continue
            value=parameters[name];expected={ParameterType.INTEGER:int,ParameterType.FLOAT:(int,float),ParameterType.BOOLEAN:bool,ParameterType.STRING:str}[d.type]
            if isinstance(value,bool) and d.type in {ParameterType.INTEGER,ParameterType.FLOAT} or not isinstance(value,expected):raise ResearchError("INVALID_STRATEGY_PARAMETER_TYPE: "+name)
        return True
    def register(self,plugin):
        self.validate(plugin);key=(plugin.metadata.strategy_id,plugin.metadata.strategy_version)
        if key in self._plugins:raise ResearchError("DUPLICATE_STRATEGY_PLUGIN")
        self._plugins[key]=plugin;return plugin
    def resolve(self,strategy_id,version=None):
        matches=[p for (sid,v),p in self._plugins.items() if sid==strategy_id and (version is None or v==version)]
        if len(matches)!=1:raise ResearchError("STRATEGY_PLUGIN_NOT_FOUND_OR_AMBIGUOUS")
        return matches[0]
    def list(self,include_test=True):return [p.metadata for p in sorted(self._plugins.values(),key=lambda p:(p.metadata.strategy_id,p.metadata.strategy_version)) if include_test or not p.metadata.synthetic_test_plugin]
    @staticmethod
    def code_fingerprint(plugin):
        try:source=inspect.getsource(plugin.__class__)
        except Exception:
            path=Path(inspect.getfile(plugin.__class__));source=sha256_file(path)
        return sha256_canonical({"implementation":source,"plugin_version":plugin.metadata.plugin_version})
    def fingerprint(self,plugin,parameters):
        self.validate_parameters(plugin,parameters);code=self.code_fingerprint(plugin);param=sha256_canonical(parameters)
        return sha256_canonical({"metadata":plugin.metadata.model_dump(mode="json"),"code_fingerprint":code,"parameter_fingerprint":param}),code,param

def default_registry():
    from .examples import SyntheticEveryNthBar
    from .s001 import S001Strategy
    registry=StrategyRegistry();registry.register(SyntheticEveryNthBar());registry.register(S001Strategy());return registry
