from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / 'artifacts/e5-reduction-blocks-local-v3-20260920'
CORE = '8b991fd7baaa470c45285754b20696c463dedc890a7db23dd4f0b9c7c818ccf1'
PROBE = 'fc0049adfdc9ac6fa18db147721e14f53e4c3b74ae142dcb7d2835801e15ed55'
LOCAL_RECEIPT = '9edf4918dcad89109a51b2bcba804a64b881306f09391d709a4c8e1faa29dde8'
LIMITS = dict(seconds=180, rss=2*1024**3, available_memory=1024**3)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path)
    with path.open('rb') as stream: return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def cases():
    rows = [(m, n, 96, False) for m in list(range(8,46))+[64,128,512] for n in [1,127,128,129,255,256,257,383,384,385,513]]
    rows += [(m, n, k, False) for m in [8,30,128,512] for n,k in [(384,384),(384,1536),(1536,384)]]
    rows += [(m, 385, 192, True) for m in [8,13,14,15,27,28,29,30,42,128]]
    assert len(rows) == 473
    return rows


def validate_resources(identity, samples):
    assert identity['complete'] and identity['code'] == 0 and not identity.get('error') and identity['limits'] == LIMITS
    assert [r['name'] for r in identity['runs']] == ['plain','code']
    previous = identity['started']; births = {(identity['supervisor']['pid'], identity['supervisor']['start'])}; result=[]
    for run in identity['runs']:
        assert run['code'] == 0 and previous <= run['started'] <= run['ended'] <= identity['ended']; previous=run['ended']
        assert 0 < run['seconds'] < LIMITS['seconds'] and run['preflight_available'] >= 4*1024**3
        assert run['preflight_disk'] >= 128*1024**2 and run['start'] >= identity['supervisor']['start']
        assert run['members'][str(run['pid'])] == run['start']
        births.add((run['pid'],run['start'])); births.update((int(pid),birth) for pid,birth in run['members'].items())
        rows=samples[run['name']]; assert len(rows)==run['samples'] and rows
        prior=0.; peak=0
        for row in rows:
            assert prior <= row['seconds'] <= run['seconds'] and row['seconds']-prior < 10; prior=row['seconds']
            assert row['available_memory'] >= LIMITS['available_memory']
            size=sum(m['rss'] for m in row['members']); assert size < LIMITS['rss']; peak=max(peak,size)
            assert len({m['pid'] for m in row['members']}) == len(row['members'])
            for member in row['members']:
                assert run['members'][str(member['pid'])] == member['start'] and member['group'] == run['pid']
                assert member['affinity'] == '2' and member['rss'] >= 0 and member['cpu_seconds'] >= 0
        assert run['seconds']-prior < 10 and peak == run['peak_rss']
        result.append(dict(name=run['name'],seconds=run['seconds'],samples=len(rows),peak_rss=peak))
    return dict(births=[dict(pid=pid,start=birth) for pid,birth in sorted(births)],runs=result)
