from types import SimpleNamespace

from sandbox.mt5 import MT5Adapter


class FakeMT5:
    TIMEFRAME_M1 = 1
    def initialize(self, **kwargs): return True
    def shutdown(self): pass
    def last_error(self): return (0, "ok")
    def terminal_info(self): return SimpleNamespace(name="Fake", connected=True)
    def account_info(self): return SimpleNamespace(company="IC Markets", server="ICMarkets-Demo", currency="USD")
    def symbols_get(self): return [SimpleNamespace(name="EURUSD.a", description="Euro vs US Dollar", digits=5, point=.00001, trade_tick_size=.00001, trade_contract_size=100000, visible=True, trade_mode=4)]


def test_read_only_adapter_resolves_broker_suffix():
    adapter = MT5Adapter(module=FakeMT5()); status = adapter.connect()
    assert status.server == "ICMarkets-Demo"
    assert adapter.resolve_symbol("EURUSD").broker_symbol == "EURUSD.a"
    forbidden = {"order_send", "positions_close", "order_modify", "trade"}
    assert not forbidden.intersection(dir(adapter))
    adapter.shutdown()

