"""Lossless native hourly EURUSD BI5 acquisition, restricted to 2016-2024."""
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path
import struct
import tempfile
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo
from sandbox.provenance import sha256_file

UTC = timezone.utc
START = datetime(2016, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 1, tzinfo=UTC)
ROOT = Path('data/external/dukascopy/EURUSD')
BASE = 'https://datafeed.dukascopy.com/datafeed/EURUSD/'
VERSION = 'DUKASCOPY_EURUSD_TICKS_V1'
RECORD = struct.Struct('>IIIff')
SCALE = 100000
EA = Path('reference_strategies/F1_NY3H_Green_MT5_Version0.mq5')
EA_SHA = '80736cd520f5a28836bdf271e5a01036f1e2c3fb4f208e42d612f32ada08ce05'
MAX_COMPRESSED = 32 * 1024 * 1024
MAX_DECODED = 128 * 1024 * 1024


def hour_utc(hour):
    if hour.tzinfo is None:
        raise ValueError('aware timestamp required')
    hour = hour.astimezone(UTC)
    if not START <= hour < END or hour.minute or hour.second or hour.microsecond:
        raise ValueError('only whole EURUSD hours in 2016-2024 permitted')
    return hour


def source_url(hour):
    h = hour_utc(hour)
    return BASE + f'{h.year}/{h.month-1:02d}/{h.day:02d}/{h.hour:02d}h_ticks.bi5'


def expected_closure(hour):
    h = hour_utc(hour)
    ny = h.astimezone(ZoneInfo('America/New_York'))
    return ((ny.weekday() == 4 and ny.hour >= 17) or ny.weekday() == 5
            or (ny.weekday() == 6 and ny.hour < 17)
            or (h.month, h.day) in ((1, 1), (12, 25)))


def decode(payload, hour):
    """Hourly UTC epoch; big-endian integer prices /100000; native float volumes."""
    hour = hour_utc(hour)
    if not payload:
        if not expected_closure(hour):
            raise ValueError('unexpected empty trading-hour payload')
        return []
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE, memlimit=MAX_DECODED)
    try:
        raw = decoder.decompress(payload, max_length=MAX_DECODED+1)
    except lzma.LZMAError as exc:
        raise ValueError('corrupt BI5 LZMA stream') from exc
    if len(raw) > MAX_DECODED or not decoder.eof or decoder.unused_data:
        raise ValueError('incomplete, oversized or trailing BI5 stream')
    if len(raw) % RECORD.size:
        raise ValueError('malformed 20-byte tick record')
    if not raw and not expected_closure(hour):
        raise ValueError('unexpected empty trading-hour archive')
    ticks = []
    previous = -1
    for ms, ask, bid, ask_vol, bid_vol in RECORD.iter_unpack(raw):
        if ms >= 3600000:
            raise ValueError('timestamp outside expected hourly partition')
        if ms < previous:
            raise ValueError('backwards source timestamps; refusing sort')
        if not 10000 <= bid <= 500000 or not 10000 <= ask <= 500000:
            raise ValueError('invalid EURUSD price scaling/range')
        if ask < bid:
            raise ValueError('Ask below Bid')
        if not all(math.isfinite(v) and v >= 0 for v in (ask_vol, bid_vol)):
            raise ValueError('invalid quote volume')
        ticks.append((hour+timedelta(milliseconds=ms), bid, ask, bid_vol, ask_vol))
        previous = ms
    return ticks


def price(integer):
    return f'{integer//SCALE}.{integer%SCALE:05d}'


def conversion(ticks):
    decoded = io.StringIO(newline='')
    mt5 = io.StringIO(newline='')
    d = csv.writer(decoded, lineterminator='\n')
    m = csv.writer(mt5, lineterminator='\n')
    d.writerow(['timestamp_utc', 'source_sequence', 'bid', 'ask', 'bid_volume', 'ask_volume'])
    m.writerow(['<DATE>', '<TIME>', '<BID>', '<ASK>', '<LAST>', '<VOLUME>'])
    for seq, (stamp, bid, ask, bv, av) in enumerate(ticks):
        d.writerow([stamp.isoformat(timespec='milliseconds'), seq, price(bid), price(ask), repr(bv), repr(av)])
        m.writerow([stamp.strftime('%Y.%m.%d'), stamp.strftime('%H:%M:%S.')+f'{stamp.microsecond//1000:03d}', price(bid), price(ask), '', ''])
    a, b = decoded.getvalue().encode(), mt5.getvalue().encode()
    verify_conversion(ticks, a, b)
    return a, b


def verify_conversion(ticks, decoded, mt5):
    rows = list(csv.reader(io.StringIO(mt5.decode())))[1:]
    details = list(csv.DictReader(io.StringIO(decoded.decode())))
    if len(rows) != len(ticks) or len(details) != len(ticks):
        raise ValueError('converted tick-count mismatch')
    for seq, (tick, row, detail) in enumerate(zip(ticks, rows, details)):
        stamp, bid, ask, bv, av = tick
        expected = [stamp.strftime('%Y.%m.%d'), stamp.strftime('%H:%M:%S.')+f'{stamp.microsecond//1000:03d}', price(bid), price(ask), '', '']
        if (row != expected or int(detail['source_sequence']) != seq
                or detail['timestamp_utc'] != stamp.isoformat(timespec='milliseconds')
                or detail['bid'] != price(bid) or detail['ask'] != price(ask)
                or float(detail['bid_volume']) != bv or float(detail['ask_volume']) != av):
            raise ValueError('conversion modified source observations/order')


def publish(path, data):
    """Atomic create-only publication; preserve existing files and interrupted sources."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'existing artifact conflict: {path}')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f'concurrent artifact conflict: {path}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch(hour):
    url = source_url(hour)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={
                'User-Agent': 'TradingResearchSandbox/1.0', 'Accept-Encoding': 'identity'})
            with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
                body = response.read(MAX_COMPRESSED+1)
                expected = response.headers.get('Content-Length')
                if response.status != 200:
                    raise ValueError(f'HTTP {response.status}')
                if len(body) > MAX_COMPRESSED or (expected is not None and int(expected) != len(body)):
                    raise ValueError('incomplete/oversized HTTP body')
                return 200, body
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and expected_closure(hour):
                return 404, None
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            retry = exc.headers.get('Retry-After', '0')
            # Non-numeric Retry-After dates are not ignored: stop for manual cooldown.
            if not retry.isdigit():
                raise RuntimeError(f'provider requested cooldown until {retry}') from exc
            delay = max(2**(attempt+1), int(retry))
            if delay > 60:
                raise RuntimeError(f'provider rate limit requires {delay}s cooldown') from exc
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**(attempt+1))


def acquire_hour(root, hour, fetcher=fetch):
    hour = hour_utc(hour)
    root = Path(root)
    key = hour.strftime('%Y/%m/%d/%H')
    raw = root/'raw'/f'{key}h_ticks.bi5'
    decoded_path = root/'decoded'/f'{key}.csv'
    mt5_path = root/'mt5'/f'{key}.csv'
    manifest = root/'manifests'/f'{key}.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        checksum = record.pop('manifest_payload_sha256')
        if hashlib.sha256(json_bytes(record)).hexdigest() != checksum:
            raise ValueError('manifest checksum mismatch')
        record['manifest_payload_sha256'] = checksum
        if (record['pipeline_version'] != VERSION or record['partition'] != hour.isoformat()
                or record['source_url'] != source_url(hour)):
            raise ValueError('manifest identity mismatch')
        paths = ((raw, 'source_filename', 'source_sha256'),
                 (decoded_path, 'decoded_filename', 'decoded_sha256'),
                 (mt5_path, 'converted_filename', 'converted_sha256'))
        for path, name, digest in paths:
            if record[name] is not None:
                if record[name] != str(path.relative_to(root)) or sha256_file(path) != record[digest]:
                    raise ValueError('cached artifact checksum/path mismatch')
            elif name != 'source_filename' or record['http_status'] != 404 or not expected_closure(hour):
                raise ValueError('missing cached artifact')
        if record['validation_status'] not in ('VALID', 'EXPECTED_CLOSURE_NO_ARCHIVE') or record['conversion_status'] != 'VALID':
            raise ValueError('unvalidated cached manifest')
        if record['source_tick_count'] != record['converted_tick_count']:
            raise ValueError('cached count mismatch')
        return record
    # A raw orphan left by interrupted conversion is revalidated, never overwritten.
    status, payload = (200, raw.read_bytes()) if raw.exists() else fetcher(hour)
    if status == 404:
        if not expected_closure(hour) or payload is not None:
            raise ValueError('unconfirmed missing trading-hour archive')
        ticks = []
    elif status == 200:
        publish(raw, payload)
        ticks = decode(payload, hour)
    else:
        raise ValueError(f'unsuccessful acquisition HTTP {status}')
    decoded, converted = conversion(ticks)
    publish(decoded_path, decoded)
    publish(mt5_path, converted)
    spreads = [ask-bid for _, bid, ask, _, _ in ticks]
    duplicates = sum(ticks[i][0] == ticks[i-1][0] for i in range(1, len(ticks)))
    record = {
        'pipeline_version': VERSION, 'provider': 'Dukascopy', 'symbol': 'EURUSD',
        'source_format': 'hourly LZMA-alone BI5 >IIIff', 'source_timezone': 'UTC',
        'normalized_timezone': 'UTC', 'price_divisor': SCALE, 'partition': hour.isoformat(),
        'source_url': source_url(hour), 'http_status': status,
        'source_filename': str(raw.relative_to(root)) if status == 200 else None,
        'source_sha256': sha256_file(raw) if status == 200 else None, 'source_tick_count': len(ticks),
        'decoded_filename': str(decoded_path.relative_to(root)), 'decoded_sha256': sha256_file(decoded_path),
        'converted_filename': str(mt5_path.relative_to(root)), 'converted_sha256': sha256_file(mt5_path),
        'converted_tick_count': len(ticks),
        'first_timestamp': ticks[0][0].isoformat(timespec='milliseconds') if ticks else None,
        'last_timestamp': ticks[-1][0].isoformat(timespec='milliseconds') if ticks else None,
        'duplicate_timestamp_count': duplicates, 'backwards_timestamp_count': 0,
        'ask_below_bid_count': 0, 'malformed_record_count': 0,
        'spread_integer_sum': sum(spreads), 'spread_min': min(spreads)/SCALE if spreads else None,
        'spread_max': max(spreads)/SCALE if spreads else None,
        'validation_status': 'VALID' if status == 200 else 'EXPECTED_CLOSURE_NO_ARCHIVE',
        'conversion_status': 'VALID',
        'empty_period_basis': ('HTTP 404 plus declared FX closure calendar' if status == 404
                               else 'HTTP 200 archive with zero ticks') if not ticks else None,
    }
    record['manifest_payload_sha256'] = hashlib.sha256(json_bytes(record)).hexdigest()
    publish(manifest, json_bytes(record))
    return record


def acquire_month(root, start, fetcher=fetch, *, workers=1):
    start = hour_utc(start)
    if start.day != 1 or start.hour:
        raise ValueError('month start required')
    if workers not in (1, 2, 3, 4):
        raise ValueError('bounded concurrency must be 1 through 4')
    end = datetime(start.year+int(start.month == 12), start.month % 12+1, 1, tzinfo=UTC)
    records = []
    def acquire_record(hour):
        try:
            return acquire_hour(root, hour, fetcher)
        except Exception as exc:
            raw = Path(root)/'raw'/f'{hour:%Y/%m/%d/%H}h_ticks.bi5'
            failure = {
                'pipeline_version': VERSION, 'provider': 'Dukascopy', 'symbol': 'EURUSD',
                'partition': hour.isoformat(), 'source_url': source_url(hour),
                'source_filename': str(raw.relative_to(root)) if raw.exists() else None,
                'source_sha256': sha256_file(raw) if raw.exists() else None,
                'source_tick_count': None, 'converted_tick_count': None,
                'validation_status': 'FAILED', 'conversion_status': 'NOT_CERTIFIED',
                'error_type': type(exc).__name__, 'error': str(exc),
                'http_status': getattr(exc, 'code', None),
            }
            encoded = json_bytes(failure)
            publish(Path(root)/'manifests'/'failures'/(hashlib.sha256(encoded).hexdigest()+'.json'), encoded)
            raise
    hour = start
    # Fixed batches bound both in-flight requests and hourly decoded memory.
    # Results are consumed in source partition order regardless of completion order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while hour < end:
            batch = []
            for _ in range(workers):
                if hour >= end:
                    break
                batch.append(pool.submit(acquire_record, hour))
                hour += timedelta(hours=1)
            records.extend(task.result() for task in batch)
    count = sum(x['source_tick_count'] for x in records)
    if not count:
        raise ValueError('month has no ticks; cannot certify acquisition')
    nonempty = [x for x in records if x['source_tick_count']]
    summary = {
        'month': start.strftime('%Y-%m'), 'validation_status': 'VALID', 'hours': len(records), 'ticks': count,
        'first_timestamp': nonempty[0]['first_timestamp'], 'last_timestamp': nonempty[-1]['last_timestamp'],
        'duplicate_timestamp_count': sum(x['duplicate_timestamp_count'] for x in records),
        'spread_min': min(x['spread_min'] for x in nonempty), 'spread_max': max(x['spread_max'] for x in nonempty),
        'spread_mean': sum(x['spread_integer_sum'] for x in records)/count/SCALE,
        'hour_manifest_checksums': [x['manifest_payload_sha256'] for x in records],
    }
    publish(Path(root)/'manifests'/f'{start:%Y-%m}.json', json_bytes(summary))
    print(json.dumps(summary | {'hour_manifest_checksums': 'stored in month manifest'}), flush=True)
    return summary


def verify_frozen():
    from sandbox.market_state.discovery.validation import load_frozen_inputs, no_fitting
    with no_fitting():
        load_frozen_inputs()
    if sha256_file(EA) != EA_SHA:
        raise ValueError('canonical Version 0 SHA256 mismatch')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--end-exclusive', default='2025-01', help='YYYY-MM; January 2016 proof always first')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    end = datetime.strptime(args.end_exclusive, '%Y-%m').replace(tzinfo=UTC)
    if not datetime(2016, 2, 1, tzinfo=UTC) <= end <= END:
        parser.error('end must be 2016-02 through 2025-01')
    verify_frozen()
    month = START
    try:
        while month < end:
            acquire_month(args.root, month, workers=args.workers)
            month = datetime(month.year+int(month.month == 12), month.month % 12+1, 1, tzinfo=UTC)
    except Exception as exc:
        failure = {'status': 'BLOCKED_NOT_COMPLETE', 'month': month.strftime('%Y-%m'),
                   'error_type': type(exc).__name__, 'error': str(exc)}
        encoded = json_bytes(failure)
        publish(args.root/'manifests'/'failures'/(hashlib.sha256(encoded).hexdigest()+'.json'), encoded)
        print(json.dumps(failure), flush=True)
        raise SystemExit(1)
    verify_frozen()


if __name__ == '__main__':
    main()
