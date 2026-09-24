"""Audit raw perf records and attribute samples only inside measured native requests."""
from bisect import bisect_right
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import struct
from run import APP, ROOT, pin, read, write

BASE = ROOT/'artifacts/parakeet-ort-native-samples-20260924'


def raw_records(path):
    counts = Counter(); lost = 0; addresses = {}
    with path.open('rb') as stream:
        assert stream.read(8) == b'PERFILE2'
        stream.seek(0); header = struct.unpack('<9Q', stream.read(72))
        start, size = header[5:7]; end = start+size
        assert 72 <= start < end <= path.stat().st_size
        stream.seek(start)
        while stream.tell() < end:
            offset = stream.tell(); kind, misc, length = struct.unpack('<IHH', stream.read(8))
            assert length >= 8 and offset+length <= end
            payload = stream.read(length-8); counts[kind] += 1
            if kind == 9:
                ip,pid,tid,stamp = struct.unpack_from('<QIIQ',payload)
                key = (pid,tid,stamp); assert key not in addresses
                addresses[key] = ip
            # Linux perf_event.h: LOST has id,lost; LOST_SAMPLES has lost.
            if kind == 2: lost += struct.unpack_from('<Q', payload, 8)[0]
            if kind == 13: lost += struct.unpack_from('<Q', payload, 0)[0]
        assert stream.tell() == end
    return dict(counts=dict(counts), lost=lost, data_start=start, data_bytes=size), addresses


def samples(path):
    header = re.compile(r'^\s*(\d+)/(\d+)\s+(\d+)\.(\d{9}):\s+(\d+)\s*$')
    frame = re.compile(r'^\s*([0-9a-f]+)\s+\((.*)\)$')
    current = None
    with path.open(encoding='utf8') as stream:
        for line in stream:
            if not line.strip(): continue
            h = header.match(line)
            if h:
                if current is not None:
                    yield current
                current = dict(pid=int(h[1]), tid=int(h[2]), stamp=int(h[3])*10**9+int(h[4]),
                               period=int(h[5]), frames=[])
                continue
            f = frame.match(line)
            assert f is not None and current is not None, line
            current['frames'].append(dict(ip=int(f[1],16), dso=f[2]))
    assert current is not None; yield current


def main():
    assert not (BASE/'closed.json').exists()
    terminal = read(BASE/'terminal.json'); state = terminal['state']; folder = BASE/'collected'
    assert state['complete'] and state['code'] == 0
    transfer = read(BASE/'transfer.json')
    assert transfer['passed'] and transfer['terminal'] == pin(BASE/'terminal.json') and transfer['archive'] == pin(BASE/'results.tar.gz')
    for name,wanted in terminal['files'].items(): assert pin(folder/name) == wanted
    limits = read(BASE/'prepared.json')['limits']; preflight = read(folder/'preflight.json')
    assert preflight['available'] >= limits['preflight_available'] and preflight['tmpfs'] >= limits['preflight_tmpfs']
    rows = [json.loads(s) for s in (folder/'resources.jsonl').read_text().splitlines()]
    assert len(rows) == state['samples'] and rows
    for row in rows:
        assert row['seconds'] < limits['seconds'] and row['rss'] < limits['rss']
        assert row['available'] >= limits['available'] and row['tmpfs'] >= limits['tmpfs'] and row['output'] < limits['output']
        assert sum(m['rss'] for m in row['members']) == row['rss']
        for member in row['members']:
            assert state['members'][str(member['pid'])] == member['birth']
            assert member['affinity'] in [[0],[2]] and all(a == member['affinity'] for a in member['threads'])
            if member['native']:
                assert member['affinity'] == [2] and {k:member[k] for k in ['pid','birth']} == state['target']
    assert all(0 <= b['seconds']-a['seconds'] < 10 for a,b in zip(rows,rows[1:]))
    module = importlib.util.spec_from_file_location('native_protocol', APP/'collected/runtime/protocol.py')
    protocol = importlib.util.module_from_spec(module); module.loader.exec_module(protocol)
    result = read(folder/'requests/result.json'); manifest = read(APP/'collected/manifests/current-parakeet.json')
    protocol.validate_records(result,manifest,'timing')
    payload = read(APP/'payload.json')
    assert result['runner_sha256'] == pin(APP/'collected/runtime/native.py')['sha256']
    assert result['manifest_sha256'] == pin(APP/'collected/manifests/current-parakeet.json')['sha256']
    assert result['adapter_sha256'] == manifest['adapter']['sha256'] and result['python_binary'] == payload['interpreter']
    assert result['native_binaries'] == manifest['native_binaries'] and result['versions'] == manifest['native_versions']
    for path,wanted in result['numeric_libraries'].items(): assert payload['external'][path] == wanted
    for index,row in enumerate(result['records']): assert read(folder/'requests'/f'{index:03}.json') == row
    module = importlib.util.spec_from_file_location('cpu_accounting',APP/'collected/runtime/campaign_processes.py')
    account = importlib.util.module_from_spec(module); module.loader.exec_module(account)
    assert state['accounting'] == account.foreign_fraction(read(folder/'cpu-before.json'),read(folder/'cpu-after.json'),state['supervisor']['pid'])
    assert state['accounting']['valid'] and state['accounting']['foreign_cpu_fraction'] <= .01
    raw,addresses = raw_records(folder/'perf.data'); assert raw['lost'] == 0
    assert not (folder/'export.stderr').read_text().strip()
    inspected = read(BASE/'native-inspection.json'); assert inspected['perf_stats']['code'] == 0
    assert '# clockid: monotonic (1)' in inspected['perf_stats']['stdout']
    assert payload['external'][inspected['path']] == inspected['binary']
    match_base = ROOT/'artifacts/parakeet-ort-native-kernel-match-20260924'
    match = read(match_base/'analysis.json'); assert match['passed'] and read(match_base/'closed.json')['passed']
    assert read(match_base/'prepared.json')['binary'] == inspected['binary']
    assert match['candidate_sha256'] == match['original_sha256'] == pin(match_base/'original.bin')['sha256']
    assert (match_base/'original.bin').read_bytes() == (match_base/'candidate.bin').read_bytes()
    maps=[]
    for line in (folder/'target.maps').read_text().splitlines():
        p=line.split(maxsplit=5)
        if len(p)==6:
            a,b=p[0].split('-');maps.append(dict(start=int(a,16),end=int(b,16),offset=int(p[2],16),path=p[5],perms=p[1]))
    windows = sorted((r['start_ticks'],r['end_ticks']) for r in result['records'] if r['phase']=='measured')
    assert len(windows)==60 and all(a[1]<=b[0] for a,b in zip(windows,windows[1:]))
    starts=[a for a,b in windows]; total=measured=weight=kernel_weight=0; dsos=Counter();ips=Counter();per_request=Counter()
    threads=Counter(); missing_stacks=[]; seen=set()
    for sample in samples(folder/'perf.script'):
        key=(sample['pid'],sample['tid'],sample['stamp']); assert key in addresses and key not in seen;seen.add(key)
        total+=1;index=bisect_right(starts,sample['stamp'])-1
        if index<0 or sample['stamp']>=windows[index][1]:continue
        assert sample['pid']==state['target']['pid']
        measured+=1;weight+=sample['period'];per_request[index]+=sample['period']
        threads[sample['tid']]+=sample['period']
        ip=addresses[key]
        if sample['frames']:
            leaf=sample['frames'][0];assert leaf['ip']==ip
        else:
            mapped=[m for m in maps if m['start']<=ip<m['end'] and 'x' in m['perms']]
            assert len(mapped)<=1
            leaf=dict(ip=ip,dso=mapped[0]['path'] if mapped else '[unresolved]')
            missing_stacks.append(dict(pid=sample['pid'],tid=sample['tid'],stamp=sample['stamp'],period=sample['period'],raw_leaf=leaf))
        dsos[leaf['dso']]+=sample['period']
        if leaf['dso']==inspected['path']:
            mapped=[m for m in maps if m['start']<=leaf['ip']<m['end'] and m['path']==leaf['dso'] and 'x' in m['perms']]
            assert len(mapped)==1
            offset=leaf['ip']-mapped[0]['start']+mapped[0]['offset'];ips[offset]+=sample['period']
            if match['start']<=offset<match['end']:kernel_weight+=sample['period']
    assert total==raw['counts'][9] and seen==set(addresses) and set(per_request)==set(range(60))
    wall=sum(b-a for a,b in windows)
    assert abs(weight-wall)/wall<=.05
    corpus=wall/3e9
    control=read(ROOT/'artifacts/parakeet-ort-diagnosis-amd-20260924/analysis.json')['phases']['control']['corpus_seconds']
    analysis=dict(passed=True,raw=raw,total_samples=total,measured_samples=measured,measured_period_ns=weight,
        measured_wall_ns=wall,coverage_ratio=weight/wall,corpus_seconds=corpus,over_control=corpus/control,
        kernel=dict(name='MlasGemmFloatKernelAvx512F',start=hex(match['start']),end=hex(match['end']),
                    matched_bytes=match['bytes'],sha256=match['candidate_sha256'],sample_period_ns=kernel_weight,
                    share=kernel_weight/weight,estimated_seconds_per_corpus=kernel_weight/3e9),
        dsos=[dict(path=p,period_ns=w,share=w/weight) for p,w in dsos.most_common()],
        instruction_addresses=[dict(offset=hex(p),period_ns=w) for p,w in ips.most_common()],
        per_request_period_ns=dict(per_request),resource_samples=len(rows),peak_rss=max(r['rss'] for r in rows),
        thread_period_ns=dict(threads),missing_stacks=missing_stacks,
        auxiliary_thread_period_ns=sum(w for tid,w in threads.items() if tid!=state['target']['pid']),
        original_consumer_unchanged=True,loaded_binary=inspected['binary'],kernel_match=pin(match_base/'closed.json'),
        attribution_only=True)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),transfer=pin(BASE/'transfer.json'),
        preparation=pin(BASE/'prepared.json'),inspection=pin(BASE/'native-inspection.json'),kernel_match=pin(match_base/'closed.json'),
        auditor=pin(__file__),audit_review=pin(BASE/'audit-review.json'),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for p,b in state['members'].items()]))
    print(json.dumps({k:analysis[k] for k in ['passed','total_samples','measured_samples','coverage_ratio','corpus_seconds','over_control','kernel','peak_rss']}))


if __name__=='__main__':main()
