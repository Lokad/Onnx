"""Publish only a complete, audited matched AMD Whisper application comparison."""
import datetime,json,shutil
from common import *

def main():
    audit=read(BASE/'audit.json');assert audit['passed'] and audit['counts']['measured']==240
    base=BASE/'collected';receipt=read(base/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    ssh(PRELUDE+'terminal(%r)\n'%receipt['births'])
    folder=Path(__file__).parent;report=folder/'results-20260921.md';data=folder/'observations-20260921.json';write(data,audit)
    header='| Workload | Lokad seconds | Microsoft ORT seconds | Lokad / ORT | Lokad RTF | ORT RTF |\n|---|---:|---:|---:|---:|---:|'
    lines=[]
    for r in audit['table']:
        name='Whisper Large V3 Turbo, all 20 clips' if r['name']=='complete-corpus' else r['name']
        lines.append(f"| {name} | {r['managed']['seconds']:.3f} | {r['ort']['seconds']:.3f} | {r['ratio']:.3f} | {r['managed']['rtf']:.3f} | {r['ort']['rtf']:.3f} |")
    headline=header+'\n'+lines[0];row=audit['table'][0];variation=[]
    for engine in ['managed','ort']:
        v=row[engine]['visits'];variation.append(f"| {engine} | {v[0]['mean']:.6f} | {v[1]['mean']:.6f} |")
    resources='\n'.join(f"| {o['name']} | {o['resource']['samples']} | {o['resource']['peak_rss']:,} | {o['resource']['min_available']:,} | {o['value']['setup_seconds']:.3f} |" for o in audit['observations'])
    text=f'''# Matched AMD Whisper Large V3 Turbo baseline

The integrated Whisper memory implementation completes the full comparison on
AMD EPYC 9V74, logical CPU 2, .NET 10.0.8 and Microsoft ONNX Runtime 1.29.0.
Product source is `{audit['product_source'][:7]}`. The timing consumer uses the
actual assemblies from the source-equivalent [qualified archive](../../whisper/memory-product-v2/results-20260921.md).
The complete workload is twenty clips totaling {row['audio_seconds']:.3f} seconds.

{headline}

Lower is faster; Lokad/ORT above one means Lokad takes longer. The headline is
the mean total for the entire corpus. Real-time factor (RTF) divides processing
time by audio duration. These are descriptive observations from two fresh
processes per engine, without a calibrated confidence or parity claim.

Four fresh processes run in managed, ORT, ORT, managed order. Each runs one full
warmup and three measured passes. All 320 requests are retained: 80 warmup and
240 measured. Twenty fresh managed conformance requests pass first. The twenty
earlier native conformance requests are reused after verifying their complete
records, frontend arrays, models, PCM, code, libraries and resource samples.
There is no outlier deletion or replacement of measured calls.

The timer covers the complete public call from PCM through features, neural
inference, greedy decoding and owned output construction. Model setup, file access
and external validation are excluded. Both engines pad each input to the same
thirty-second encoder extent. ORT uses a pinned Transformers NumPy frontend and
CPUExecutionProvider, one intra/inter-op thread, sequential execution, all graph
optimizations and spinning disabled. Every sampled inference thread runs on CPU 2.
Managed runs use normal collection, no forced GC and no LOKAD/DOTNET/COMPlus overrides.

Every request passes exact text, token, stop and no-speech decisions, input
preservation, held actual-output ownership and within-engine repeat checks.
Whisper confidence remains exact across managed repeats. Native recomputed
features satisfy the original 1e-5 frontend gate. These application checks do not
close the documented full encoder/logit numerical failures.

## Process variation

| Engine | First process corpus mean seconds | Second process corpus mean seconds |
|---|---:|---:|
{chr(10).join(variation)}

## Resource observations

| Worker | Samples | Peak RSS bytes | Minimum available bytes | Setup seconds |
|---|---:|---:|---:|---:|
{resources}

Every worker passes the unchanged 3,600-second / 14 GiB RSS / 1 GiB available / 32 MiB disk
guards, with 13 GiB available and 64 MiB disk before launch. The campaign is bounded
at 14,400 seconds. Raw samples retain all process creation times, thread affinities,
available RAM, disk and sample gaps. Setup, allocation, GC and foreign CPU
observations remain diagnostics; none is subtracted from timed requests.

The [original all-family memory failure](../amd-comparison/resource-failure-20260920.md)
remains valid. This separate comparison measures the now-integrated context reuse
and exact private decoder-weight sharing. Earlier Windows and AMD tables use
different revisions/protocols; their absolute times do not establish a speedup
from this change. A prelaunch runtime-path assertion also remains recorded: no
inference started until the actual qualified host and runtime files were bound.

## Every clip

{header}
{chr(10).join(lines[1:])}

## Evidence

All 340 new requests and 20 referenced native requests pass the complete audit.
Forty-five damaged coverage, identity and resource records are rejected. All
actual process identities are terminal before collection and reporting.
Frozen SHA256:`{audit['frozen']['sha256']}`. The collection binds
{len(receipt['files'])} files and verifies {receipt['external_verified']} external
identities. Raw artifacts are under artifacts/audio-whisper-amd-20260921;
[complete observations](observations-20260921.json) retain unrounded values.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    benchmark=ROOT/'BENCHMARK.md';s=benchmark.read_text(encoding='utf-8')
    start=s.index('**Whisper has no AMD timing result:**');end=s.index('### Audio: Windows Microsoft ONNX Runtime baselines',start)
    section=f'''### Audio: matched AMD Whisper baseline

AMD EPYC 9V74, logical CPU 2, .NET 10.0.8, Microsoft ORT 1.29.0, product `{audit['product_source'][:7]}`.
The complete workload is twenty clips ({row['audio_seconds']:.3f} seconds of audio).
Times include frontend, neural inference, decoding and owned results; model loading,
file access and external validation are excluded.

{headline}

Two fresh processes per engine each run one warmup and three measured passes;
all 240 measured and 80 warmup requests pass application, ownership and resource checks.
The [complete report](tests/audio/whisper-amd/results-20260921.md) includes every
clip, process variation and memory. These are descriptive results with no calibrated
parity claim. The earlier [resource failure](tests/audio/amd-comparison/resource-failure-20260920.md)
is preserved; this revision integrates qualified execution-buffer reuse and decoder
weight sharing. Existing encoder/logit numerical limits remain open. Different
revisions and hosts prevent a speedup claim against the other tables.

'''
    s=s[:start]+section+s[end:];s=s.replace('## Current results — 2026-09-20 UTC','## Current results — 2026-09-21 UTC');benchmark.write_text(s,encoding='utf-8')
    support=ROOT/'docs/model-support.md';s=support.read_text(encoding='utf-8')
    s=s.replace('Whisper has no matched AMD timing result because its managed conformance worker failed the available-memory guard.',
        'A later [matched AMD Whisper comparison](../tests/audio/whisper-amd/results-20260921.md) measures the integrated memory implementation; the original managed conformance resource failure remains recorded.')
    s=s.replace('[matched AMD Whisper comparison](../tests/audio/whisper-amd/README.md) is running;', '[matched AMD Whisper comparison](../tests/audio/whisper-amd/results-20260921.md) is complete;')
    s+='\nThe integrated Whisper memory implementation now completes the [matched AMD ORT comparison](../tests/audio/whisper-amd/results-20260921.md), including all twenty fresh managed conformance and 320 timing-stage requests. These descriptive latency results retain the existing numerical and broader accuracy limitations.\n';support.write_text(s,encoding='utf-8')
    snapshots=BASE/'closure-snapshots';snapshots.mkdir()
    for p in [benchmark,support,*sorted(folder.glob('*.py')),ROOT/'tests/audio/amd-comparison/protocol.py',ROOT/'tests/audio/amd-comparison/audit.py']:
        target=snapshots/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data]})
    write(BASE/'closed.json',dict(passed=True,files=files,births=receipt['births'],closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(passed=True,closure=pin(BASE/'closed.json'),pins=len(files),headline=row)))

if __name__=='__main__':main()
