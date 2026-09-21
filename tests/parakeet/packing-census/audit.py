"""Join actual prepared mappings to independently qualified wall observations."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
from run import ROOT, BASE, TRACE, pin, read, save, psutil


def main():
    assert not (BASE/'closed.json').exists()
    prepared = read(BASE/'prepared.json'); assert prepared['passed']
    for name, wanted in prepared['files'].items(): assert pin(ROOT/name) == wanted, name
    state = read(BASE/'state.json'); assert state['complete'] and state['passed'] and state['code'] == 0
    for key in ('supervisor', 'worker'):
        identity = state[key]
        try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess: pass
    samples = [json.loads(s) for s in (BASE/'samples.jsonl').read_text().splitlines()]
    assert len(samples) == state['samples'] > 0 and max(s['rss'] for s in samples) == state['peak_rss']
    assert state['preflight_available'] >= 10*1024**3
    assert all(s['seconds'] < 180 and s['rss'] < 8*1024**3 and s['available'] >= 1024**3
               and s['affinity'] == [2] and s['disk'] > 20*1024**3 for s in samples)
    result = read(BASE/'result.json'); assert result['passed'] and result['runtime'] == '.NET 10.0.12'
    assert result['affinity'] == 4 and result['processor_count'] == 1
    for key, filename in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','Census.dll')]:
        assert result[key] == pin(BASE/'bin'/filename)['sha256']
    trace_closed = read(TRACE/'closed.json')
    trace_graphs = read(TRACE/'trace-output/graphs.json')
    for name in ('analysis.json','trace-output/graphs.json','trace-output/result.json'):
        assert pin(TRACE/name) == trace_closed['files'][(TRACE/name).relative_to(ROOT).as_posix()]
    metadata = {}; by_shape = defaultdict(lambda:dict(nodes=0,weights=0,bytes=0,packed_weights=0,packed_bytes=0,seconds=Fraction(),packed_seconds=Fraction()))
    mappings = {}; graph_summary = {}
    for graph, info in result['graphs'].items():
        assert info['nodes'] == trace_graphs[graph]
        packed = {p['source']:p for p in info['packed_weights']}; assert len(packed) == len(info['packed_weights'])
        assert sum(p['bytes'] for p in packed.values()) == info['retained_packed_bytes'] <= info['maximum_packed_bytes']
        graph_summary[graph] = {k:info[k] for k in ('maximum_packed_bytes','retained_packed_bytes')}
        graph_summary[graph]['packed_weights'] = len(packed)
        weights = set()
        for node in info['nodes']:
            metadata[(graph,node['id'])] = node
            if node['op'] != 'MatMul': continue
            source = node['inputs'][1]; initializer = node['initializers'].get(source)
            shape = tuple(initializer['shape']) if initializer else ()
            key = graph,shape; cell = by_shape[key]; cell['nodes'] += 1
            mappings[(graph,node['id'])] = source in packed
            if initializer and source not in weights:
                weights.add(source); assert initializer['dtype'] == 'Float' and len(shape) == 2
                size = 4*shape[0]*shape[1]; cell['weights'] += 1; cell['bytes'] += size
                if source in packed:
                    assert packed[source]['shape'] == list(shape) and packed[source]['bytes'] == size
                    cell['packed_weights'] += 1; cell['packed_bytes'] += size
        assert set(packed) <= weights
    trace_result = read(TRACE/'trace-output/result.json')
    for name in trace_result['call_files'][1240:]:
        path = TRACE/'trace-output'/name
        assert pin(path) == trace_closed['files'][path.relative_to(ROOT).as_posix()]
        row = read(path); assert row['pass'] == 1
        for item in row['nodes']:
            if item['op'] != 'MatMul': continue
            node = metadata[(row['graph'],item['id'])]; init = node['initializers'].get(node['inputs'][1])
            shape = tuple(init['shape']) if init else (); cell = by_shape[(row['graph'],shape)]
            seconds = Fraction(item['end_ticks']-item['start_ticks'],row['frequency']); cell['seconds'] += seconds
            if mappings[(row['graph'],item['id'])]: cell['packed_seconds'] += seconds
    profile = read(TRACE/'analysis.json')
    for graph in result['graphs']:
        total = sum(v['seconds'] for (g,s),v in by_shape.items() if g == graph)
        assert float(total) == next(v['seconds'] for v in profile['operators'] if v['graph'] == graph and v['op'] == 'MatMul')
    rows = [dict(graph=g,shape=list(shape),**{k:float(v) if isinstance(v,Fraction) else v for k,v in cell.items()})
            for (g,shape),cell in sorted(by_shape.items())]
    summary = dict(passed=True,graphs=graph_summary,matmul=rows,
        scope='Actual prepared mapping presence joined to profiled node cost; mapping presence does not prove packed kernel dispatch')
    save(BASE/'analysis.json',summary)
    files = dict(prepared['files'])
    for folder in (BASE,Path(__file__).parent,ROOT/'artifacts/parakeet-matmul-source-20260921'):
        for p in folder.rglob('*'):
            if p.is_file() and not {'obj','packages'}.intersection(p.parts): files[p.relative_to(ROOT).as_posix()] = pin(p)
    save(BASE/'closed.json',dict(passed=True,files=files,identities=[state[k] for k in ('supervisor','worker')]))
    print(json.dumps(summary)); print(json.dumps(dict(closed=pin(BASE/'closed.json'),files=len(files))))


if __name__ == '__main__': main()
