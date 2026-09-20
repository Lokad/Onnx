"""Write complete AMD Parakeet/pyannote results, preserving failed Whisper scope."""
import datetime,json,shutil
from pathlib import Path
from stage import BASE,ROOT,ssh
from protocol import pin,read,write


def main():
    audit=read(BASE/'audit.json');assert audit['passed'] and audit['counts']==dict(referenced_conformance=48,timing=384,warmup=96,measured=288)
    base=BASE/'collected';receipt=read(base/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    response=ssh('''import sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
print('terminal')
'''%receipt['births']);assert response.strip()=='terminal'
    folder=Path(__file__).resolve().parent;report=folder/'results-20260920.md';data=folder/'observations-20260920.json'
    write(data,audit)
    header='| Application / workload | Lokad seconds | Microsoft ORT seconds | Lokad / ORT | Lokad RTF | ORT RTF |\n|---|---:|---:|---:|---:|---:|'
    headline=[];details=[];variation=[]
    for row in audit['table']:
        name=('Parakeet' if row['family']=='parakeet' else 'pyannote')+', '+('all 20 clips (213.265 s audio)' if row['name']=='complete-corpus' else row['name'])
        line=f"| {name} | {row['managed']['seconds']:.3f} | {row['ort']['seconds']:.3f} | {row['ratio']:.3f} | {row['managed']['rtf']:.3f} | {row['ort']['rtf']:.3f} |"
        if row['name']=='complete-corpus' or row['family']=='pyannote':
            headline.append(line)
            for engine in ['managed','ort']:
                v=row[engine]['visits'];variation.append(f"| {name} | {engine} | {v[0]['mean']:.6f} | {v[1]['mean']:.6f} |")
        else:details.append(line)
    table=header+'\n'+'\n'.join(headline)
    resource='\n'.join(f"| {o['name']} | {o['resource']['samples']} | {o['resource']['peak_rss']:,} | {o['value']['setup_seconds']:.3f} |" for o in audit['observations'])
    text=f'''# Matched AMD Parakeet and pyannote application baselines

All eight timing workers complete the fixed comparison on AMD EPYC 9V74,
logical CPU 2, .NET 10.0.8 and Microsoft ONNX Runtime 1.29.0. Product source is
`1d10d22`, including the qualified WeSpeaker frontend correction. Exact previously
qualified source-archive binaries and unchanged application consumers are used.
The Windows results have their own host and revision; comparing absolute times
across the two tables does not establish a product performance change.

{table}

Lower is faster; Lokad/ORT above one means Lokad takes longer. Parakeet's headline
is the mean total for all twenty clips. Pyannote times are per request; the three
ten-second crops overlap the full dialogue. Real-time factor (RTF) is processing
time divided by audio duration. These are descriptive observations from two fresh
processes per engine, without calibrated confidence intervals or a parity claim.

The order is Parakeet managed, ORT, ORT, managed, then pyannote in the same order.
Each process runs one complete warmup and three measured passes. All 384 requests
are retained: 96 warmup and 288 measured, with no outlier deletion or partial corpus.
The gate reuses 48 complete conformance requests under identical binaries, models,
manifests and runtime libraries. Those original four workers are not rerun.

The timer includes the public application call: features, neural inference,
decoding or automatic clustering, and owned output construction. Loading, file
access and external validation are excluded. Native pyannote combines ORT graphs
with pinned Torch frontend/pooling and NumPy/SciPy/upstream clustering. Both
engines receive the same FP32 models and PCM arrays.

Every request passes exact token/text/stop or diarization decision checks,
centroid scaled error at 1e-4, interval endpoints at 1e-12, input preservation,
held-output ownership and deterministic repeats. ORT uses CPUExecutionProvider,
one intra/inter-op thread, sequential execution, all graph optimizations and no
spinning. Numerical-library threads are one; every sampled worker thread remains
on CPU 2. Managed runs have no LOKAD, DOTNET or COMPlus overrides.

Whisper has **no matched AMD timing result**. The separate [all-family campaign](../amd-comparison/resource-failure-20260920.md)
failed its original available-memory reserve during managed Whisper conformance,
before timing. Native Whisper's 20 calls and managed Whisper's 16 saved requests
passed their checks, but managed completion was not established. This two-family
scope was declared before collecting any timing data. It preserves that failure
and does not relax its resource limits or qualify repeated Whisper use on this VM.
The complete historical Windows Whisper baseline remains in BENCHMARK.md.

## Process variation

| Workload | Engine | First process mean seconds | Second process mean seconds |
|---|---|---:|---:|
{chr(10).join(variation)}

## Resource observations

| Worker | Samples | Peak RSS bytes | Setup seconds |
|---|---:|---:|---:|
{resource}

Every process remains within the original 3,600-second, 14 GiB RSS, 1 GiB available-memory
and 32 MiB disk limits. Resource logs include all members, births, thread affinities,
available RAM, disk and timestamps. Foreign CPU accounting remains in the raw
observations. Setup and GC/allocation observations are additional diagnostics,
not subtracted from public application times.

## Per-clip Parakeet results

{header}
{chr(10).join(details)}

## Evidence

The independent audit verifies all 384 timing and 48 referenced conformance records,
all raw request files and resource samples, unchanged product/model identities,
exact ordering and final output ownership. Seventy-two damaged coverage, identity
and resource records are rejected. All original new-campaign process births are
absent before collection and again before reporting.

Artifact: artifacts/audio-amd-two-family-20260920. Frozen manifest SHA-256:
`{pin(base/'frozen.json')['sha256']}`. Collection binds {len(receipt['files'])} files
and verifies {receipt['external_verified']:,} external model/library/evidence files.
Archive SHA-256:`{read(BASE/'collection-transfer.json')['archive']['sha256']}`.
Full observations: [JSON](observations-20260920.json). This application comparison
does not close the separately documented Whisper encoder/logit, segmentation
silence, direct-native frontend or broader speech/diarization accuracy limits.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    benchmark=ROOT/'BENCHMARK.md';current=benchmark.read_text(encoding='utf-8')
    old='''The Windows comparisons below include explicit ORT baselines for both Parakeet
and pyannote. The separate September 20 AMD all-family campaign stopped during
managed Whisper conformance at its available-memory reserve, before timing.
Parakeet and pyannote passed both engines' conformance checks; a separate matched
AMD timing comparison is being prepared. The [resource-failure report](tests/audio/amd-comparison/resource-failure-20260920.md)
retains the incomplete Whisper run. No AMD latency is inferred from those checks.

'''
    assert current.count(old)==1;current=current.replace(old,'')
    marker='### Audio: matched Microsoft ONNX Runtime baselines';assert current.count(marker)==1
    section=f'''### Audio: matched AMD Parakeet and pyannote baselines

AMD EPYC 9V74, logical CPU 2, .NET 10.0.8, Microsoft ORT 1.29.0, product `1d10d22`.
The same complete application workloads and one-thread settings as the Windows
table below are used. Loading, file access and external validation are excluded.
The differing host and product revision prevent a cross-table speedup claim.

{table}

Two fresh processes per engine/model each run one full warmup and three measured
passes. All 288 measured and 96 warmup calls pass application, ownership, input and
resource checks. Forty-eight complete conformance calls are reused after verifying
unchanged models, binaries and runtime libraries. Lower is faster; ratios above
one mean Lokad takes longer. These descriptive results have no calibrated parity
claim. The [complete report](tests/audio/amd-two-family/results-20260920.md)
includes process variation, memory, every Parakeet clip and evidence identities.

**Whisper has no AMD timing result:** the separate [all-family attempt](tests/audio/amd-comparison/resource-failure-20260920.md)
stopped during managed conformance below its 1 GiB available-memory reserve, with
16 of 20 requests completed, before timing. The Windows Whisper baseline remains
below. The later two-family timing scope preserves that resource failure.

'''
    benchmark.write_text(current.replace(marker,section+'### Audio: Windows Microsoft ONNX Runtime baselines').replace(
        'The audio timing tables remain measurements of `8732831`.',
        'The Windows audio tables measure `8732831`; the AMD Parakeet/pyannote table measures `1d10d22`.'),encoding='utf-8')
    support=ROOT/'docs/model-support.md';content=support.read_text(encoding='utf-8')
    content=content.replace('The audio timing tables remain measurements of `8732831`.',
        'The Windows audio tables measure `8732831`. The [matched AMD Parakeet/pyannote comparison](../tests/audio/amd-two-family/results-20260920.md) measures `1d10d22` against ORT 1.29.0. Whisper has no matched AMD timing result because its managed conformance worker failed the available-memory guard.')
    support.write_text(content,encoding='utf-8')
    snapshots=BASE/'closure-snapshots';snapshots.mkdir()
    for p in [benchmark,support,ROOT/'tests/audio/amd-comparison/audit.py',ROOT/'tests/audio/amd-comparison/protocol.py']:
        target=snapshots/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data,*sorted(folder.glob('*.py'))]})
    write(BASE/'closed.json',dict(passed=True,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,births=receipt['births']))
    print(json.dumps(dict(passed=True,receipt=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
