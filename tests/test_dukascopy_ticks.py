import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import lzma
from types import SimpleNamespace
import urllib.error

import pytest
from sandbox.market_data import dukascopy_ticks as d

HOUR = datetime(2016, 1, 4, 12, tzinfo=timezone.utc)


def payload(records=None):
    if records is None:
        records = [(123, 108720, 108714, 1.25, 2.5),
                   (123, 108721, 108715, 2.5, 1.25),
                   (123, 108721, 108715, 2.5, 1.25),
                   (3599999, 108725, 108724, 0., 0.)]
    return lzma.compress(b''.join(d.RECORD.pack(*row) for row in records), format=lzma.FORMAT_ALONE)


def test_native_layout_scaling_mapping_milliseconds_and_utc():
    ticks = d.decode(payload(), HOUR)
    stamp, bid, ask, bv, av = ticks[0]
    assert stamp == HOUR + timedelta(milliseconds=123)
    assert stamp.utcoffset() == timedelta(0)
    assert d.price(bid) == '1.08714' and d.price(ask) == '1.08720'
    assert bv == 2.5 and av == 1.25
    assert ticks[-1][0] == HOUR + timedelta(milliseconds=3599999)
    assert d.source_url(HOUR).endswith('EURUSD/2016/00/04/12h_ticks.bi5')


def test_duplicates_and_source_order_are_preserved_exactly():
    ticks = d.decode(payload(), HOUR)
    assert len(ticks) == 4 and ticks[1] == ticks[2]
    decoded, mt5 = d.conversion(ticks)
    rows = list(csv.DictReader(io.StringIO(decoded.decode())))
    assert [r['source_sequence'] for r in rows] == ['0', '1', '2', '3']
    assert [r['bid'] for r in rows[:3]] == ['1.08714', '1.08715', '1.08715']
    output = list(csv.reader(io.StringIO(mt5.decode())))
    assert output[1] == ['2016.01.04', '12:00:00.123', '1.08714', '1.08720', '', '']
    assert output[2] == output[3]


def test_deterministic_conversion_and_hashing():
    ticks = d.decode(payload(), HOUR)
    assert d.conversion(ticks) == d.conversion(ticks)
    a, b = d.conversion(ticks)
    assert hashlib.sha256(b).digest() == hashlib.sha256(d.conversion(ticks)[1]).digest()
    d.verify_conversion(ticks, a, b)


@pytest.mark.parametrize('bad', [b'html error', payload()[:-5], payload()+b'trailing',
    lzma.compress(b'bad record', format=lzma.FORMAT_ALONE)])
def test_corrupt_truncated_malformed_or_trailing_archive_rejected(bad):
    with pytest.raises(ValueError): d.decode(bad, HOUR)


@pytest.mark.parametrize('record', [(3600000, 108720, 108714, 1., 1.),
    (0, 108710, 108714, 1., 1.), (0, 1, 1, 1., 1.), (0, 10872000, 10871400, 1., 1.),
    (0, 108720, 108714, float('nan'), 1.), (0, 108720, 108714, float('inf'), 1.),
    (0, 108720, 108714, -1., 1.)])
def test_invalid_fields_fail_closed(record):
    with pytest.raises(ValueError): d.decode(payload([record]), HOUR)


def test_backwards_timestamps_not_sorted_or_discarded():
    with pytest.raises(ValueError, match='backwards'):
        d.decode(payload([(2,108720,108714,1.,1.), (1,108720,108714,1.,1.)]), HOUR)


def test_empty_closure_is_distinct_from_missing_or_empty_trading_data(tmp_path):
    saturday = datetime(2016,1,2,12,tzinfo=timezone.utc)
    assert d.decode(b'', saturday) == []
    assert d.decode(lzma.compress(b'',format=lzma.FORMAT_ALONE), saturday) == []
    with pytest.raises(ValueError, match='empty'): d.decode(b'', HOUR)
    record = d.acquire_hour(tmp_path,saturday,lambda h:(404,None))
    assert record['validation_status'] == 'EXPECTED_CLOSURE_NO_ARCHIVE'
    assert record['source_filename'] is None and record['source_sha256'] is None
    assert record['converted_tick_count'] == 0
    with pytest.raises(ValueError): d.acquire_hour(tmp_path,HOUR,lambda h:(404,None))
    with pytest.raises(ValueError): d.acquire_hour(tmp_path,HOUR,lambda h:(429,None))


def test_conversion_loss_or_change_rejected():
    ticks=d.decode(payload(),HOUR);a,b=d.conversion(ticks)
    with pytest.raises(ValueError,match='count'):
        d.verify_conversion(ticks,a,b'\n'.join(b.splitlines()[:-1])+b'\n')
    with pytest.raises(ValueError,match='modified'):
        d.verify_conversion(ticks,a,b.replace(b'1.08714',b'1.08713'))


def test_resume_skips_validated_source_and_rechecks_hashes(tmp_path):
    calls=[]
    def fetch(hour): calls.append(hour);return 200,payload()
    first=d.acquire_hour(tmp_path,HOUR,fetch)
    assert first==d.acquire_hour(tmp_path,HOUR,fetch)
    assert calls==[HOUR]
    assert first['source_tick_count']==first['converted_tick_count']==4
    assert first['duplicate_timestamp_count']==2
    (tmp_path/first['converted_filename']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):d.acquire_hour(tmp_path,HOUR,fetch)
    assert calls==[HOUR]


def test_interrupted_conversion_resumes_from_preserved_source(tmp_path,monkeypatch):
    original=d.conversion
    monkeypatch.setattr(d,'conversion',lambda ticks:(_ for _ in ()).throw(RuntimeError('interrupted')))
    with pytest.raises(RuntimeError):d.acquire_hour(tmp_path,HOUR,lambda h:(200,payload()))
    assert list((tmp_path/'raw').rglob('*.bi5'))
    assert not list((tmp_path/'manifests').rglob('*.json'))
    monkeypatch.setattr(d,'conversion',original)
    record=d.acquire_hour(tmp_path,HOUR,lambda h:pytest.fail('must reuse raw'))
    assert record['validation_status']=='VALID'


def test_corrupt_source_preserved_never_marked_valid(tmp_path):
    with pytest.raises(ValueError):d.acquire_hour(tmp_path,HOUR,lambda h:(200,b'bad'))
    source=next((tmp_path/'raw').rglob('*.bi5'))
    assert source.read_bytes()==b'bad'
    with pytest.raises(ValueError):d.acquire_hour(tmp_path,HOUR,lambda h:pytest.fail('must not overwrite'))


@pytest.mark.parametrize('hour',[datetime(2025,1,1,tzinfo=timezone.utc), datetime(2015,12,31,23,tzinfo=timezone.utc),
    datetime(2016,1,4,12), HOUR+timedelta(milliseconds=1)])
def test_protected_or_invalid_request_rejected_before_network(tmp_path,hour):
    with pytest.raises(ValueError):d.acquire_hour(tmp_path,hour,lambda h:pytest.fail('network forbidden'))


def test_utc_normalization():
    other=HOUR.astimezone(timezone(timedelta(hours=2)))
    assert d.source_url(other)==d.source_url(HOUR)
    assert d.decode(payload(),other)==d.decode(payload(),HOUR)


def test_manifest_tampering_rejected(tmp_path):
    d.acquire_hour(tmp_path,HOUR,lambda h:(200,payload()))
    p=next((tmp_path/'manifests').rglob('*.json'))
    record=json.loads(p.read_text());record['source_tick_count']=99;p.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='manifest checksum'):d.acquire_hour(tmp_path,HOUR)


def test_provider_429_retried_bounded_never_closure(monkeypatch):
    calls=[];sleeps=[]
    def fail(req,**kw):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url,429,'Too Many Requests',{},None)
    monkeypatch.setattr(d.urllib.request,'build_opener',lambda *a:SimpleNamespace(open=fail))
    monkeypatch.setattr(d.time,'sleep',sleeps.append)
    with pytest.raises(urllib.error.HTTPError):d.fetch(d.START)
    assert len(calls)==3 and sleeps==[2,4]
    assert all('/2016/00/01/' in url for url in calls)


def test_january_failure_blocks_bulk_and_writes_failure_manifest(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(d,'verify_frozen',lambda:None)
    def fail(root,start,**kwargs):calls.append(start);raise ValueError('January failed')
    monkeypatch.setattr(d,'acquire_month',fail)
    with pytest.raises(SystemExit):d.main(['--root',str(tmp_path)])
    assert calls==[d.START]
    record=json.loads(next((tmp_path/'manifests'/'failures').glob('*.json')).read_text())
    assert record['status']=='BLOCKED_NOT_COMPLETE'


def test_month_covers_all_hours_without_crossing_final_holdout(tmp_path,monkeypatch):
    hours=[]
    def fake(root,hour,fetcher):
        hours.append(hour)
        return {'source_tick_count':1,'first_timestamp':hour.isoformat(),'last_timestamp':hour.isoformat(),
            'duplicate_timestamp_count':0,'spread_min':.0001,'spread_max':.0001,'spread_integer_sum':10,
            'manifest_payload_sha256':'a'*64}
    monkeypatch.setattr(d,'acquire_hour',fake)
    result=d.acquire_month(tmp_path,datetime(2024,12,1,tzinfo=timezone.utc))
    assert result['hours']==result['ticks']==744
    assert hours[-1]==datetime(2024,12,31,23,tzinfo=timezone.utc)
    assert all(h<d.END for h in hours)


def test_canonical_ea_and_frozen_model_contract_hashes():
    d.verify_frozen()
    assert d.sha256_file(d.EA)==d.EA_SHA


def test_existing_artifact_conflict_never_overwritten(tmp_path):
    p=tmp_path/'x';d.publish(p,b'original')
    with pytest.raises(ValueError):d.publish(p,b'new')
    assert p.read_bytes()==b'original'


def test_parallel_failure_stops_after_bounded_batch(tmp_path,monkeypatch):
    from threading import Barrier, Lock
    barrier=Barrier(4); lock=Lock(); seen=[]
    def fail(root,hour,fetcher):
        with lock: seen.append(hour)
        barrier.wait(timeout=10)
        raise ValueError('provider unavailable')
    monkeypatch.setattr(d,'acquire_hour',fail)
    with pytest.raises(ValueError,match='provider unavailable'):
        d.acquire_month(tmp_path,d.START,workers=4)
    assert len(seen)==4
    assert sorted(seen)==[d.START+timedelta(hours=i) for i in range(4)]
    assert not (tmp_path/'manifests'/'2016-01.json').exists()


def test_parallel_month_summary_still_chronological(tmp_path,monkeypatch):
    def fake(root,hour,fetcher):
        return {'source_tick_count':1,'first_timestamp':hour.isoformat(),'last_timestamp':hour.isoformat(),
            'duplicate_timestamp_count':0,'spread_min':.0001,'spread_max':.0001,'spread_integer_sum':10,
            'manifest_payload_sha256':hour.isoformat()}
    monkeypatch.setattr(d,'acquire_hour',fake)
    result=d.acquire_month(tmp_path,d.START,workers=4)
    assert result['ticks']==744
    assert result['hour_manifest_checksums']==sorted(result['hour_manifest_checksums'])
