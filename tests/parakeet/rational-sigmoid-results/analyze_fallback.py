"""Summarize closed fallback evidence without trimming or timing admission."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-rational-sigmoid-fallback-diagnostic-amd-20260927'
METHOD = re.compile(r'^; Assembly listing for method ([^\n]+) \(([^\n]+)\)\n(.*?); Total bytes of code (\d+)[^\n]*', re.M | re.S)
INSTRUCTION = re.compile(r'^\s+([0-9A-F]{2,30})\s+(.+)$', re.M)


def pin(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def analyze():
    closure = read(BASE/'closed.json')
    assert closure['passed'] and not closure['admitted']
    assert closure['analysis'] == pin(BASE/'analysis.json')
    analysis = read(BASE/'analysis.json')
    assert analysis['diagnostic_only'] and not analysis['performance_admission_attempted']
    folder = BASE/'capture-collected'
    assert closure['collection'] == pin(folder/'capture-collection.json')
    collection = read(folder/'capture-collection.json')
    processes = {}
    for name in ['current-0', 'candidate-1', 'candidate-2', 'current-3']:
        path = folder/'logs'/(name+'.json')
        assert pin(path) == analysis['outputs'][name]
        raw = read(path)
        cases = []
        for row in raw['rows']:
            groups = {False: [], True: []}
            for c, o in zip(row['clocks'][600:], row['observations'][600:], strict=True):
                assert c['iteration'] == o['iteration'] and not c['warmup']
                groups[any(o[k] for k in ['gen0', 'gen1', 'gen2'])].append((c, o))
            assert sum(map(len, groups.values())) == 180
            parts = {}
            for changed, records in groups.items():
                ticks = sum(c['ticks'] for c, _ in records)
                calls = len(records)*row['batch']
                parts['collection_counter_changed' if changed else 'no_collection_counter_change'] = dict(
                    batches=len(records), calls=calls, total_ticks=ticks,
                    seconds_per_call=None if not calls else float(Fraction(ticks, calls*raw['frequency'])),
                    allocated_bytes=sum(o['allocated'] for _, o in records),
                    collections={k: sum(o[k] for _, o in records) for k in ['gen0', 'gen1', 'gen2']})
            index = row['index']
            cases.append(dict(name=row['name'], all_seconds_per_call=analysis['instrumented_case_seconds'][name][index],
                allocation=analysis['observations'][name]['cases'][index], partition=parts))
        assembly = folder/'logs'/(name+'.stdout')
        assert pin(assembly) == collection['files']['logs/'+name+'.stdout']
        matches = list(METHOD.finditer(assembly.read_text(encoding='utf8')))
        assert len(matches) == len(analysis['codegen'][name]['listings'])
        scalar = []
        for match in matches:
            if match[2] != 'Tier1' or 'CPUExecutionProvider:Sigmoid(' not in match[1]: continue
            for block in re.split(r'(?=^G_M\d+_IG\d+:)', match[3], flags=re.M):
                if not re.search(r'call\s+System.MathF?:Exp\(', block): continue
                ins = [m[2].strip() for m in INSTRUCTION.finditer(block)]
                scalar.append(dict(dtype='float' if 'System.MathF:Exp' in block else 'double',
                    assembly=block, instructions=ins, opcodes=[s.split()[0] for s in ins],
                    stack_operands=[s for s in ins if re.search(r'\[(?:r|e)(?:sp|bp)[+\-]', s)]))
        assert scalar, 'Final optimized public wrapper was not emitted'
        processes[name] = dict(vector_width=raw['vector_width'], raw_result=pin(path), raw_assembly=pin(assembly),
            cases=cases, optimized_scalar_blocks=scalar,
            whole_run_allocated_bytes=sum(o['allocated'] for r in raw['rows'] for o in r['observations']),
            whole_run_collections={k:sum(o[k] for r in raw['rows'] for o in r['observations']) for k in ['gen0','gen1','gen2']},
            listings=[dict(signature=m[1], tier=m[2], native_bytes=int(m[4])) for m in matches])
    return dict(diagnostic_only=True, no_admission=True, no_sample_removed=True,
        closure=pin(BASE/'closed.json'), analysis=pin(BASE/'analysis.json'),
        products=analysis['products'], samples=analysis['samples'], resources=analysis['resources'], processes=processes,
        limitations=['Counters bracket a slightly larger interval than the timer.',
            'Groups are an exhaustive diagnostic partition, not corrected timing or admission.',
            'No allocation-latency or tier-to-clock measurement.',
            'New diagnostic events do not identify historical GC or tiers.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true'); args = parser.parse_args()
    result = analyze()
    path = Path(__file__).with_name('fallback-observations-20260927.json')
    text = json.dumps(result, indent=2) + '\n'
    if args.publish:
        with path.open('x', encoding='utf8', newline='\n') as stream: stream.write(text)
    elif path.exists(): assert path.read_text(encoding='utf8') == text
    for name, p in result['processes'].items():
        print(json.dumps(dict(process=name, vector_width=p['vector_width'],
            scalar_blocks=[dict(dtype=b['dtype'], instructions=len(b['instructions']), stack_operands=len(b['stack_operands'])) for b in p['optimized_scalar_blocks']],
            cases=[r for r in p['cases'] if r['name'] in ['scalar-option', 'double', 'empty', 'scalar']])))
