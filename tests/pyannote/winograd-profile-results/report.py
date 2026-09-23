"""Publish every qualified diagnostic observation without a speed-selection claim."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-winograd-profile-amd-20260923'
OUT = Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def csv_file(name, rows):
    with (OUT/name).open('x',encoding='utf8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    assert not (OUT/'results-20260923.md').exists()
    proof = read(BASE/'closed.json'); assert proof['passed']
    for name,wanted in proof['files'].items(): assert pin(ROOT/name) == wanted,name
    value = read(BASE/'analysis.json'); assert value['passed'] and proof['analysis'] == pin(BASE/'analysis.json')
    assert value['calls'] == 48 and value['measured_calls'] == 36 and value['exact_prior_amd_results']
    assert [d['name'] for d in value['diagnostics']] == ['sampled-a','sampled-b']
    clocks = []; setup = []
    for process in ['control','sampled-a','sampled-b']:
        result = read(BASE/'collected'/process/'result.json')
        setup.append(dict(process=process,seconds=result['setup_seconds']))
        for row in result['records']:
            clocks.append(dict(process=process,**{key:row[key] for key in ['name','pass','phase',
                'start_ticks','end_ticks','frequency','seconds','cpu_user_ticks','cpu_system_ticks',
                'cpu_frequency','allocated_bytes','thread_id']}))
    assert len(clocks) == 48 and sum(r['phase']=='measured' for r in clocks) == 36
    csv_file('clocks-20260923.csv',clocks); csv_file('setup-20260923.csv',setup)
    stacks = []; leaves = {}
    for capture in value['diagnostics']:
        totals = capture['selected_seconds']; by_method = defaultdict(float)
        for kind in ['exclusive','inclusive']:
            for row in capture[kind]:
                stacks.append(dict(capture=capture['name'],kind=kind,marker=row['marker'],method=row['method'],
                    bucket=row.get('bucket',''),seconds=row['seconds'],share=row['seconds']/totals[row['marker']]))
                if kind == 'exclusive' and row['marker']=='dialogue-30s': by_method[row['method']] += row['seconds']
        assert abs(sum(by_method.values())/totals['dialogue-30s']-1)<1e-6
        leaves[capture['name']] = {method:seconds/totals['dialogue-30s'] for method,seconds in by_method.items()}
    csv_file('stacks-20260923.csv',stacks)
    methods = set().union(*(set(rows) for rows in leaves.values()))
    ranked = sorted(methods,key=lambda m:sum(rows.get(m,0) for rows in leaves.values()),reverse=True)
    resources = value['resources']; samples = sum(r['samples'] for r in resources)
    peak = max(r['peak_rss'] for r in resources)
    lines = ['# Current Winograd Pyannote attribution', '',
        'Both captures pass the complete diagnostic audit for integrated M34',
        'Core `521bae17` / Data `f3b9aa81`. These are sampled request-thread',
        'weights, affected by inlining and profiler overhead. Wall and process CPU',
        'are independently measured below. No ORT process is timed here and this',
        'diagnostic does not establish a product speedup.', '',
        '| Exclusive sampled leaf, complete dialogue | Capture A | Capture B |',
        '|---|---:|---:|']
    for method in ranked[:16]:
        label = method.replace('|','\\|')
        lines.append(f"| {label} | {100*leaves['sampled-a'].get(method,0):.3f}% | {100*leaves['sampled-b'].get(method,0):.3f}% |")
    lines += ['', 'The sixteen largest leaves are ranked by their combined share across',
        'both captures. [Every exclusive and inclusive stack](stacks-20260923.csv)',
        'for all four fixtures is retained. Inclusive stacks contain their children;',
        'do not add them to exclusive percentages.', '',
        '| Fixture | Process | Wall seconds | Process CPU seconds | Allocated bytes | Wall / control |',
        '|---|---|---:|---:|---:|---:|']
    for row in value['observations']:
        for process,clock in row['roles'].items():
            ratio = 1 if process=='control' else row['diagnostic_to_control'][process]
            lines.append(f"| {row['name']} | {process} | {clock['wall_mean']:.9f} | {clock['process_cpu_mean']:.9f} | {clock['allocated_mean']:.3f} | {ratio:.6f} |")
    lines += ['', 'Each process performs four warmups and twelve measured requests:',
        '**48 complete public results / 36 measured requests**. Every result exactly',
        'matches the admitted current-product application reference. All original',
        'native bounds, input preservation and returned-output ownership checks pass.',
        '[Every raw wall/CPU clock](clocks-20260923.csv), including warmups, and',
        '[all setup intervals](setup-20260923.csv) are retained. The table uses',
        'three measured requests per fixture and process; no clock is excluded.', '',
        'Both Speedscope/Chromium exports reconcile. All measured marker intervals',
        'belong to the original request thread, with no warmup markers in sampled',
        'exports. The original coverage tolerances and release barriers pass.',
        f'All **{samples:,} resource observations** pass; peak owned RSS is **{peak:,} bytes**.',
        'Every build, capture and converter owner is terminal with code zero.',
        'Targets inherit CPU2 before startup; collectors/monitor/converters use CPU0.',
        'The product is copied unchanged. All160consumer methods are compared:',
        '159 unchanged, Main equal after exactly two product identity substitutions.', '',
        'The matched application baseline remains **10.453276795 seconds versus',
        'Microsoft ORT9.040290939, ratio1.156299**. Its separate <=1.05 parity',
        'target remains unmet. Use this diagnostic and code inspection to choose',
        'a distinct next experiment; no application gain follows from sampling alone.', '',
        'Artifact: `artifacts/pyannote-winograd-profile-amd-20260923`.',
        'Closure (repository-relative keys):', '`'+pin(BASE/'closed.json')['sha256']+'`.',
        'Payload:', '`'+pin(BASE/'payload/payload.json')['sha256']+'`.', '']
    (OUT/'results-20260923.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')
    report = dict(closed=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),
        generator=pin(Path(__file__)),observations=value['observations'],
        diagnostics=[{k:v for k,v in d.items() if k not in ['exclusive','inclusive']} for d in value['diagnostics']],
        resources=resources,consumer=value['consumer'],
        outputs={name:pin(OUT/name) for name in ['results-20260923.md','clocks-20260923.csv',
            'setup-20260923.csv','stacks-20260923.csv']})
    (OUT/'observations-20260923.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,calls=len(clocks),resources=samples,peak_rss=peak,report=report['outputs']['results-20260923.md'])))


if __name__ == '__main__': main()
