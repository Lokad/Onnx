"""Join every saved periodic sample, then map sigmoid addresses to emitted code."""
import base64
from bisect import bisect_right
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import struct
import subprocess
from run import ROOT, BASE, CAPTURE, EVENTS, pin, read, write, checked

TIERS = {3:'QuickJitted', 4:'OptimizedTier1', 5:'OptimizedTier1OSR', 6:'QuickJittedInstrumented'}


def rows(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def decode_method(event):
    raw = base64.b64decode(event['rawBase64'], validate=True)
    method, module, start, size, token, flags = struct.unpack_from('<QQQIII', raw)
    names = []; offset = 36
    for _ in range(3):
        end = offset
        while raw[end:end+2] != b'\0\0':
            assert end+2 <= len(raw)
            end += 2
        names.append(raw[offset:end].decode('utf-16-le')); offset = end+2
    if names[0] != 'Lokad.Onnx.CPUExecutionProvider' or names[1] not in ['Sigmoid', 'SigmoidRationalVector']:
        return None
    assert event['pointerSize'] == 8 and event['version'] in [1, 2]
    assert len(raw)-offset == (2 if event['version'] == 1 else 10)
    return dict(event_index=event['index'], ms=event['ms'], pid=event['pid'], method_id=hex(method),
        module_id=hex(module), start=start, end=start+size, size=size, token=token, flags=flags,
        tier=TIERS[(flags>>7)&7], namespace=names[0], name=names[1], signature=names[2],
        clr_instance=struct.unpack_from('<H', raw, offset)[0],
        rejit=struct.unpack_from('<Q', raw, offset+2)[0] if event['version'] == 2 else 0)


def generated_code():
    path = CAPTURE/'collected/logs/sampled-target.log'
    result = {}
    for section in path.read_text(encoding='utf8').split('; Assembly listing for method '):
        if not section.startswith('Lokad.Onnx.CPUExecutionProvider:Sigmoid') or not section.splitlines()[0].endswith('(Tier1)'):
            continue
        name = section.split('(')[0].split(':')[1]
        code = bytearray(); instructions = []
        for line in section.splitlines():
            block = re.match(r'G_M\d+_IG\d+:\s+;; offset=0x([0-9A-Fa-f]+)', line)
            if block:
                assert len(code) == int(block[1], 16), (name, line, len(code))
            raw = re.match(r'^\s+([0-9A-F]{2,})\s+(.*)$', line)
            if raw:
                data = bytes.fromhex(raw[1])
                instructions.append(dict(offset=len(code), bytes=raw[1], text=raw[2]))
                code.extend(data)
            if line.startswith('; Total bytes of code '):
                assert len(code) == int(line.split()[-1])
        assert name not in result
        result[name] = dict(bytes=len(code), instructions=instructions, data=bytes(code))
    assert {n:v['bytes'] for n,v in result.items()} == dict(Sigmoid=3240, SigmoidRationalVector=555)
    return result


def main():
    checked(); assert not (BASE/'closed.json').exists()
    folder = BASE/'collected'; receipt = read(folder/'collection.json')
    transfer = read(BASE/'transfer.json'); state = read(folder/'state.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz')
    assert transfer['collection'] == pin(folder/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and state['complete'] and state['code'] == 0
    assert state['supervisor'] == read(BASE/'deployment.json') and state['inference_calls'] == 0
    assert len(state['runs']) == 1 and state['runs'][0]['name'] == 'addresses'
    assert receipt['identities'] == [state['supervisor'], state['runs'][0]['processes']['export']]
    for name, wanted in receipt['files'].items():
        assert pin(folder/name) == wanted, name
    run = state['runs'][0]
    resources = [json.loads(line) for line in (folder/'logs/addresses.samples.jsonl').read_text().splitlines()]
    assert len(resources) == run['samples'] > 0 and run['code'] == 0 and run['complete'] and run['seconds'] < 900
    assert max(r['rss'] for r in resources) == run['peak_rss']
    for row in resources:
        assert row['seconds'] < 900 and row['rss'] < 12*1024**3
        assert min(row['available'], row['disk']) >= 1024**3 and row['output_bytes'] <= 512*1024**2
        for member in row['members']:
            assert member['pid'] == run['processes']['export']['pid'] and member['birth'] == run['processes']['export']['birth']
            assert member['affinity'] == [0] and member['threads'] and all(t['affinity'] == [0] for t in member['threads'])
    summary = read(folder/'output/summary.json')
    assert summary['passed'] and summary['lost'] == 0 and not summary['symbol_downloads'] and summary['inference_calls'] == 0
    assert summary['pid'] == run['processes']['export']['pid']
    assert summary['input_sha256'] == pin(CAPTURE/'collected/sampled/capture.nettrace')['sha256']
    assert summary['library_sha256'] == pin(EVENTS/'Microsoft.Diagnostics.Tracing.TraceEvent.dll')['sha256']
    addresses = rows(folder/'output/addresses.jsonl.gz')
    stacks = rows(folder/'output/stacks.jsonl.gz')
    samples = rows(folder/'output/samples.jsonl.gz')
    boundaries = rows(folder/'output/boundaries.jsonl.gz')
    assert summary['counts'] == dict(addresses=len(addresses), stacks=len(stacks), samples=len(samples),
        boundaries=len(boundaries), missingStacks=0, allEvents=676767)
    assert len(samples) == 115795 and len(boundaries) == 120
    assert [r['index'] for r in addresses] == list(range(len(addresses)))
    assert [r['index'] for r in stacks] == list(range(len(stacks)))
    for stack in stacks:
        assert 0 <= stack['address'] < len(addresses) and -1 <= stack['caller'] < stack['index']
    for index, sample in enumerate(samples):
        assert sample['index'] == index and 0 <= sample['stack'] < len(stacks)
    sample_index = boundary_index = total_events = 0; methods = []
    with gzip.open(CAPTURE/'collected/events/events.jsonl.gz', 'rt', encoding='utf8') as stream:
        for line in stream:
            event = json.loads(line); total_events += 1
            if event['provider'] == 'Microsoft-DotNETCore-SampleProfiler' and event['id'] == 0:
                selected = samples[sample_index]; sample_index += 1
            elif event['provider'] == 'Lokad-Pyannote-Diagnostic':
                selected = boundaries[boundary_index]; boundary_index += 1
                assert selected['id'] == event['id']
            else:
                selected = None
            if selected is not None:
                assert selected['pid'] == event['pid'] and selected['thread'] == event['thread']
                assert abs(selected['ms']-event['ms']) <= 1e-8 and selected['raw'] == event['rawBase64']
            if event['provider'] == 'Microsoft-Windows-DotNETRuntimeRundown' and event['id'] in [143, 144]:
                method = decode_method(event)
                if method is not None:
                    methods.append(method)
    assert total_events == 676767 and sample_index == len(samples) and boundary_index == len(boundaries)
    assert len(methods) == 8
    code = generated_code(); code_dir = BASE/'code'; code_dir.mkdir()
    for name, body in code.items():
        selected = [m for m in methods if m['name'] == name and m['tier'] == 'OptimizedTier1']
        assert len(selected) == 1 and selected[0]['size'] == body['bytes']
        path = code_dir/(name+'.bin'); path.write_bytes(body.pop('data'))
        command = ['C:/Strawberry/c/bin/objdump.exe', '-D', '-b', 'binary', '-m', 'i386:x86-64',
                   '--adjust-vma='+hex(selected[0]['start']), str(path)]
        result = subprocess.run(command, capture_output=True, check=True, timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
        (code_dir/(name+'.objdump.txt')).write_bytes(result.stdout)
        body.update(binary=pin(path), method=selected[0], disassembly=pin(code_dir/(name+'.objdump.txt')))
    capture_analysis = read(CAPTURE/'analysis.json')
    intervals = capture_analysis['events']['request_intervals']; begins = [r['begin_ms'] for r in intervals]
    per_request = [dict(name=r['name'], pass_index=r['pass_index'], main_samples=0, sigmoid_top=0, helper_top=0, sigmoid_ancestor=0) for r in intervals]
    mapped_addresses = {}; all_counts = Counter(); outside = Counter(); raw_kinds = Counter(); matched = []
    for index, sample in enumerate(samples):
        raw_kinds[base64.b64decode(sample['raw']).hex()] += 1
        request_index = bisect_right(begins, sample['ms'])-1
        if sample['pid'] != 1291110 or sample['thread'] != 1291110:
            outside['other_process_or_thread'] += 1; continue
        if request_index < 0 or sample['ms'] > intervals[request_index]['end_ms']:
            outside['outside_request'] += 1; continue
        counts = per_request[request_index]; counts['main_samples'] += 1
        frame_ids = []; cursor = sample['stack']
        while cursor >= 0:
            frame_ids.append(stacks[cursor]['address']); cursor = stacks[cursor]['caller']
        sigmoid_frames = [i for i in frame_ids if 'Lokad.Onnx.CPUExecutionProvider.Sigmoid' in addresses[i]['method']]
        if sigmoid_frames:
            counts['sigmoid_ancestor'] += 1
        top = frame_ids[0]; address = addresses[top]
        if top not in sigmoid_frames:
            all_counts['other_top'] += 1; continue
        candidates = [m for m in methods if m['start'] <= int(address['address'],16) < m['end']]
        assert len(candidates) == 1
        method = candidates[0]
        assert method['pid'] == sample['pid'] and method['tier'] == address['tier'] == 'OptimizedTier1'
        assert method['token'] == address['token'] and method['name'] in address['method']
        offset = int(address['address'],16)-method['start']
        instructions = code[method['name']]['instructions']
        positions = {r['offset']:i for i,r in enumerate(instructions)}
        assert offset in positions, (address, offset)
        instruction_index = positions[offset]
        key = 'helper_top' if method['name'] == 'SigmoidRationalVector' else 'sigmoid_top'
        counts[key] += 1; all_counts[key] += 1
        entry = mapped_addresses.setdefault(top, dict(address=address, method=method['name'], offset=offset,
            instruction=instructions[instruction_index], previous=instructions[instruction_index-1] if instruction_index else None,
            samples=0, requests=[]))
        entry['samples'] += 1
        if request_index not in entry['requests']:
            entry['requests'].append(request_index)
        matched.append(dict(sample=index, request=request_index, address_index=top))
    assert sum(outside.values())+sum(r['main_samples'] for r in per_request) == len(samples)
    assert sum(all_counts.values()) == sum(r['main_samples'] for r in per_request)
    assert all(r['main_samples'] > 0 for r in per_request)
    allocation_boundary = sum(r['samples'] for r in mapped_addresses.values()
        if r['method'] == 'Sigmoid' and r['previous'] and 'call' in r['previous']['text'] and 'CORINFO_HELP_NEWARR_1_VC' in r['previous']['text'])
    helper_loop = sum(r['samples'] for r in mapped_addresses.values()
        if r['method'] == 'SigmoidRationalVector' and 0x2f <= r['offset'] < 0x16b)
    assert helper_loop == all_counts['helper_top']
    upstream = read(BASE/'upstream.json')+[read(BASE/'upstream-extra.json')]
    for row in upstream:
        assert pin(BASE/row['file']) == {k:row[k] for k in ['bytes','sha256']}
    value = dict(passed=True, inference_calls=0, product_changed=False, source_closure=pin(CAPTURE/'closed.json'),
        complete_sample_join=True, periodic_samples=len(samples), request_boundaries=len(boundaries),
        all_events=total_events, counts=dict(all_counts), outside_counts=dict(outside), raw_sample_kinds=dict(raw_kinds),
        per_request=per_request, mapped_addresses=list(mapped_addresses.values()), matched_samples=matched,
        methods=methods, generated_code=code, allocation_return_boundary_samples=allocation_boundary,
        helper_vector_loop_samples=helper_loop, reader=summary, resources=dict(samples=len(resources), peak_rss=run['peak_rss']),
        observer_overhead=capture_analysis['sampled_to_control'], upstream=upstream,
        limits='Periodic managed stack samples, not instruction-cycle measurements; native callees are not exposed. No per-node join. Original thread-time export mixes allocation events with periodic samples. No overhead subtraction.')
    write(BASE/'analysis.json', value)
    inputs = {p.relative_to(ROOT).as_posix():pin(p) for p in [Path(__file__), CAPTURE/'closed.json', CAPTURE/'analysis.json', CAPTURE/'collected/logs/sampled-target.log', CAPTURE/'collected/events/events.jsonl.gz']}
    write(BASE/'closed.json', dict(passed=True, inference_calls=0, analysis=pin(BASE/'analysis.json'), inputs=inputs,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        remote_retained=receipt['retained_remote']))
    print(json.dumps(dict(passed=True, closed=pin(BASE/'closed.json'), counts=dict(all_counts), outside=dict(outside),
        allocation_boundary=allocation_boundary, helper_loop=helper_loop)))


if __name__ == '__main__':
    main()
