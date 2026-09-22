"""Publish current selected-runtime attribution after complete independent closure."""
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-selected-profile-amd-20260922'
sys.path.insert(0,str(ROOT/'tests/pyannote/selected-profile-amd'))
from common import pin, read, verify, terminal, CORE, DATA


def main():
    output = TOOLS/'results-20260922.md'; observations = TOOLS/'observations-20260922.json'
    assert not output.exists() and not observations.exists()
    closure = read(BASE/'closed.json'); assert closure['passed']; verify(closure['files'])
    for identity in closure['local_identities']: terminal(identity)
    analysis = read(BASE/'analysis.json'); assert analysis['passed'] and analysis['calls'] == 48
    assert closure['analysis'] == pin(BASE/'analysis.json')
    captures = analysis['diagnostics']; assert len(captures) == 2
    totals = []
    for capture in captures:
        weights = collections.defaultdict(float)
        for row in capture['exclusive']:
            if row['marker'] == 'dialogue-30s': weights[row['method']] += row['seconds']
        totals.append(dict(weights))
    methods = set(totals[0]) | set(totals[1]); rows = []
    for method in methods:
        values = [dict(seconds=totals[i].get(method,0),share=100*totals[i].get(method,0)/captures[i]['selected_seconds']['dialogue-30s']) for i in range(2)]
        rows.append(dict(method=method,captures=values))
    rows.sort(key=lambda r:sum(v['share'] for v in r['captures']),reverse=True)
    requests = {}
    for name in ['control','sampled-a','sampled-b']:
        result = read(BASE/'collected'/name/'result.json')
        assert result['core_sha256'] == CORE and result['data_sha256'] == DATA
        requests[name] = [{k:r[k] for k in ['name','pass','phase','start_ticks','end_ticks','frequency','seconds','cpu_user_ticks','cpu_system_ticks','cpu_frequency','thread_id','allocated_bytes']} for r in result['records']]
    value = dict(passed=True,closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),core=CORE,data=DATA,
        leaves=rows,observations=analysis['observations'],requests=requests,
        coverage=[dict(name=d['name'],coverage=d['coverage'],exports=d['exports']) for d in captures],
        resources=analysis['resources'],consumer=analysis['consumer'],
        remote_identities=analysis['remote_identities'],local_identities=analysis['local_identities'])
    observations.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    table = ['| Exclusive leaf (full request) | Capture A share | Capture B share |','|---|---:|---:|']
    for row in rows[:12]:
        method = row['method'].split('!')[-1].split('(')[0].replace('|','\\|')
        table.append(f"| `{method}` | {row['captures'][0]['share']:.2f}% | {row['captures'][1]['share']:.2f}% |")
    timing = ['| Fixture | Control mean s | Capture A mean s | Capture B mean s |','|---|---:|---:|---:|']
    for row in analysis['observations']:
        timing.append('| '+row['name']+' | '+' | '.join(f"{row['roles'][role]['wall_mean']:.6f}" for role in ['control','sampled-a','sampled-b'])+' |')
    resource_count = sum(r['samples'] for r in analysis['resources']); peak = max(r['peak_rss'] for r in analysis['resources'])
    text = [
        '# Current selected Pyannote attribution on AMD','',
        'The following table attributes the complete public dialogue request on '
        'selected Core `1279b4b6` / Data `4e602d9f`. It follows the accepted '
        'single-panel convolution change. The rejected Parakeet arithmetic '
        'candidate is not used.','',*table,'',
        'Shares are exclusive sampled managed-thread weights. Inlining may charge '
        'helper work to a caller. They do not isolate individual memory operations '
        'and are distinct from process CPU and whole-application latency. Exporter '
        'CPU_TIME labels are retained as labels.','',
        '**All 48 public requests pass**, including the original numerical, speaker, '
        'timeline, repeat, input and held-output checks. Every output exactly '
        'matches the selected AMD comparison. One control and two fresh captures '
        'each execute all four fixtures with one warmup and three measured passes. '
        'Both trace exports reconcile every event, all request markers and every '
        'fixture coverage gate.','',*timing,'',
        'These timings describe instrumentation and its control. They do not '
        'supply a new Microsoft ORT ratio or select a faster product.','',
        'The consumer changes only its two expected product hash literals: '
        '**160 existing methods, 159 unchanged**, one changed main method, '
        'unchanged public declarations and no additions/removals. The selected '
        'product binaries, Linux thread adapter, public checks, post-warmup '
        'barrier, collector and stack parsers retain their qualified bytes.','',
        f'All **{resource_count:,} resource observations** pass; peak owned RSS '
        f'is **{peak:,} bytes**. Target CPU2, collector/monitor CPU0, .NET 10.0.8 '
        'and pinned dotnet-trace 10.0.745401. Original 10 GiB target preflight, '
        '8 GiB RSS, 1 GiB available/tmpfs/output, 2 GiB experiment, 900-second '
        'pair and 180-second barrier limits remain. Local builds and exports '
        'retain their 8 GiB preflight/RSS limits. All owners are terminal.','',
        'The first audit rejected the exporter’s optional named finalizer-thread '
        'label. The additive audit_v2.py accepts that label, retains every '
        'background profile and event, and additionally rejects duplicate '
        'native thread IDs. Three parser tests pass; the same immutable traces '
        'pass all original accounting and coverage limits. No capture was rerun.','',
        'Read this alongside the [native layout diagnostic](../native-layout-amd/results-20260922.md), '
        'which confirms blocked convolutions, fused sums and activations on ORT. '
        'Any proposed optimization still needs a separate measured screen, '
        'complete numerical qualification and matched application comparison.','',
        '[Every retained leaf, request clock and coverage check](observations-20260922.json); '
        '[frozen capture protocol](../selected-profile-amd/README.md).','',
        'Artifact: `artifacts/pyannote-selected-profile-amd-20260922`.',
        f"Closure: `{pin(BASE/'closed.json')['sha256']}`.",
        f"Analysis: `{pin(BASE/'analysis.json')['sha256']}`."]
    output.write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps(dict(report=pin(output),observations=pin(observations),samples=resource_count,peak_rss=peak,top=rows[:5])))


if __name__ == '__main__': main()
