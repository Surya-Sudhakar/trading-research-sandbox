import pandas as pd

from sandbox.cli import main


def test_data_audit_parquet_uses_market_data_integrity_audit(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    parquet_path = tmp_path / "eurusd.parquet"
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"], utc=True
            ),
            "open": [1.1, 1.11],
            "high": [1.2, 1.21],
            "low": [1.0, 1.01],
            "close": [1.15, 1.16],
            "tick_volume": [1, 1],
            "spread": [1, 1],
            "real_volume": [0, 0],
        }
    ).to_parquet(parquet_path, index=False)
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SANDBOX_CATALOG_PATH", str(tmp_path / "catalog.sqlite3"))

    assert main(["data", "audit", str(parquet_path)]) == 0
    assert '"status": "clean"' in capsys.readouterr().out
