"""Close complete selected-case trace evidence without waiving any original numerical gate."""
from pathlib import Path
import argparse,datetime,importlib.util,shutil
from common import ROOT,pin,read,write,verify

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    assert not (base/'closed.json').exists();spec=read(base/'manifest.json');verify(spec);audit=read(base/'audit.json')
    assert audit['passed'] is True and audit['arrays']==656 and audit['saved_array_bytes']==5283840000
    assert audit['manifest']==pin(base/'manifest.json') and len(audit['instrumentation'])==16 and len(audit['contrasts'])==8
    for name,want in audit['files'].items():assert pin(base/name)==want,name
    loader=importlib.util.spec_from_file_location('selected_trace_checks',Path(__file__).with_name('audit.py'))
    checks=importlib.util.module_from_spec(loader);loader.loader.exec_module(checks)
    births={(r['pid'],r['birth']) for resource in audit['resources'] for r in resource['births']}
    assert all(checks.original.absent(dict(pid=pid,birth=birth)) for pid,birth in births)
    report=Path(__file__).with_name('results-20260920.md');assert not report.exists()
    lines=['# Selected natural Whisper encoder traces — 2026-09-20','',
        'The three cases nominated by the [full saved-frame analysis](../frame-distribution/results-20260920.md), plus the first-case repeat, complete on both engines and both feature sources. '
        'All eight workers, sixteen calls and 656 full arrays pass structural, input, held-output, repeat, identity and resource checks. '
        'These selected traces localize discrepancies; they do not replace full-corpus qualification.','',
        'The reused ONNX trace graph differs from the original export only in its 41 graph outputs. Protobuf comparison verifies every other field, including all 1,559 nodes and initializer/external-data declarations. '
        'Outputs expose the first convolutions/activations, positional addition, first-layer normalization/residual values, all 32 layer-end residuals and final encoder state. '
        'Exposed outputs can inhibit fusions; the following table measures each final trace array against its same-engine, same-input unmodified output from the closed corpus. No unmodified inference was rerun.','',
        '| Request | Recording | Cell | Final bits unchanged | Instrumentation max scaled | Failed values |',
        '|---:|---|---|---|---:|---:|']
    for row in audit['instrumentation']:
        lines.append(f"| {row['request']} | {row['name']} | {row['kind']} | {row['bitwise']} | {row['max_scaled']:.9g} | {row['failed_values']:,} |")
    lines+=['','MM/MN mean managed inference on managed/native features; NM/NN mean native inference on managed/native features. '
        'Each instrumentation row uses its saved unmodified same-engine array as denominator. These are measured perturbations, not corrections to the original discrepancy.','',
        'The next table compares both engines on identical features at every exposed output. Each value is maximum scaled error against the corresponding **traced native** array. '
        'The original full-corpus report used its original NN denominator for all decomposition terms; do not conflate these differently labeled comparisons. '
        'All complete per-output counts, absolute errors and L2 norms remain in audit.json.','',
        '| Output | First MF | First NF | Original-max MF | Original-max NF | Native-max MF | Native-max NF |',
        '|---|---:|---:|---:|---:|---:|---:|']
    selected=[r for r in audit['contrasts'] if r['request']<3]
    assert len(selected)==6
    for number,description in enumerate(spec['outputs']):
        lines.append('| '+description['name']+' | '+' | '.join(f"{r['outputs'][number]['max_scaled']:.7g}" for r in selected)+' |')
    lines+=['','First = `121-121726-0000`; Original-max = `1089-134686-0002`; Native-max = `908-157963-0000`. '
        'The repeated first case matches every intermediate bit exactly in all four engine/input combinations. '
        'An observed increase between boundaries does not by itself identify an incorrect operator: input propagation, conditioning and optimizer changes require a controlled follow-up. '
        'The unchanged full-array threshold is `1e-4`; no padding positions or failed values are excluded.','',
        'Workers use Windows i7-14700KF, CPU 2, original core c6bf781/.NET 10.0.12 and native ORT 1.29.0 with one intra/inter-op thread, sequential execution, all optimizations and no spinning. '
        'Managed uses fresh Memory contexts, a 256-MiB packing budget and no native ORT. The actual first-call tensors remain held through the second call and resets. '
        'Native saves its actual optimized graph and external weights for node-census verification; those serialized models are retained as evidence and are not executed on another host.','',
        f"Observed optimized nodes: managed {sum(audit['censuses']['managed'].values())}; native {sum(audit['censuses']['native'].values())}. Full per-operator censuses are retained.",'',
        'All workers satisfy the declared 1,800-second, 8-GiB sampled group-RSS and 1-GiB available-memory guards, with 10 GiB available before launch. '
        'Every observed PID/creation-time identity is terminal. Sampling does not establish an absolute peak or observe every short-lived child. No profiler, forced GC, runtime override or latency comparison is used.','']
    for engine in ['managed','native']:
        rows=[r for r in audit['resources'] if r['job']['engine']==engine]
        lines.append(f"- {engine}: {sum(r['samples'] for r in rows):,} samples; peak group RSS {max(r['peak_rss'] for r in rows):,} bytes; minimum available {min(r['minimum_available'] for r in rows):,} bytes.")
    lines+=['',f"Manifest SHA-256 `{pin(base/'manifest.json')['sha256']}`; audit SHA-256 `{pin(base/'audit.json')['sha256']}`.",
        'Full arrays (5,283,840,000 bytes), optimized native graphs, resource samples, source and closed receipts are retained in '
        '`artifacts/whisper-trace-selected-20260920`. No product/default change or general numerical qualification follows.','']
    report.write_text('\n'.join(lines),encoding='utf-8')
    snapshot=base/'closed-source';snapshot.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file():shutil.copyfile(path,snapshot/path.name)
    files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    write(base/'closed.json',dict(passed=True,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        report=dict(file=report.relative_to(ROOT).as_posix(),**pin(report)),births=[dict(pid=p,birth=b) for p,b in sorted(births)],scope=audit['scope']))
    print('Closed',len(files),'files;',pin(base/'closed.json'))

if __name__=='__main__':main()
