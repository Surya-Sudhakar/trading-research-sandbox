import argparse
import json
from pathlib import Path
from sandbox.research.registry import ResearchRegistry
from sandbox.config import Settings
from sandbox.mt5 import MT5Adapter
from .storage import sanitize_external
from .archive import MT5Archive


def main():
    p=argparse.ArgumentParser(description="Immutable multi-source market data")
    sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("sanitize");s.add_argument("raw_dataset_id")
    a=sub.add_parser("archive-once");a.add_argument("--symbol",default="EURUSD")
    a.add_argument("--initial-days",type=int,default=7);a.add_argument("--overlap-minutes",type=int,default=60)
    args=p.parse_args();settings=Settings.load()
    if args.command=="sanitize":
        registry=ResearchRegistry(settings.catalog_path)
        manifest,_=sanitize_external(registry,args.raw_dataset_id)
        print(json.dumps(manifest,indent=2))
    else:
        archive=MT5Archive(Path("data/broker_archive"),symbol=args.symbol)
        with MT5Adapter(settings.terminal_path) as adapter:
            resolved=adapter.resolve_symbol(args.symbol)
            archive.provider_symbol=resolved.broker_symbol
            # Preserve the resolved provider symbol while the archive path stays canonical.
            class Feed:
                def rates(self,symbol,start,end):return adapter.rates(resolved.broker_symbol,start,end)
            print(json.dumps(archive.poll(Feed(),initial_days=args.initial_days,overlap_minutes=args.overlap_minutes),indent=2))


if __name__=="__main__":main()
