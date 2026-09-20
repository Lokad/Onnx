"""Close the diagnostic without changing normal-runtime qualification or latency."""
import datetime,json,shutil
from pathlib import Path
from deploy import BASE,ROOT,ssh
from protocol import pin,read,write


def main():
    audit=read(BASE/'audit.json');assert audit['passed'] and audit['calls']==20 and not audit['benchmark'] and not audit['normal_runtime_qualification']
    base=BASE/'collected';receipt=read(base/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    terminal=ssh('''import sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
print('terminal')
'''%receipt['births']);assert terminal.strip()=='terminal'
    folder=Path(__file__).resolve().parent;report=folder/'results-20260920.md';data=folder/'observations-20260920.json'
    write(data,audit);lines=[]
    for row in audit['interventions']:
        a=row['before'];b=row['after']
        lines.append(f"| {row['after_call']} | {a['managed_estimate']/1e9:.6f} | {b['managed_estimate']/1e9:.6f} | {row['managed_reclaimed']/1e9:.6f} | {a['rss']/1e9:.6f} | {b['rss']/1e9:.6f} | {row['seconds']*1000:.3f} |")
    table='| After request | Heap estimate before GB | After GB | Reclaimed GB | RSS before GB | After GB | Collection ms |\n|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(lines)
    reclaimed=[r['managed_reclaimed'] for r in audit['interventions']]
    conclusion=('Each planned generation-2 collection reduces the estimated managed heap.' if all(v>0 for v in reclaimed) else 'The table retains the observed heap changes, including any collection without a reduction.')
    resource=audit['resource'];frozen=read(base/'frozen.json')
    text=f'''# Whisper memory collection diagnostic on AMD

The twenty-request diagnostic completes all original application, input and
held-output checks with explicit collections after requests 8, 16 and 20.
{conclusion} The
original normal-runtime [available-memory failure](../../audio/amd-comparison/resource-failure-20260920.md)
remains failed. This intervention is neither a production fix nor a timing result.

{table}

GB means decimal billions of bytes. Heap estimates come from
`GC.GetTotalMemory(false)` and can include garbage awaiting collection. RSS is
resident process memory, measured separately. A collection can reclaim managed
objects without immediately returning an equal number of resident bytes to the
operating system. The requested `compacting: true` does not set large-object-heap
compaction policy. The transcriber, every PCM array and all earlier actual public
result objects remain held and are rechecked across the three interventions.

All twenty transcripts, token sequences, stop and no-speech decisions match the
existing reference. The first sixteen also match the saved normal-runtime prefix
exactly. Each explicit collection advances the generation-2 count and collector
index; no inference occurs inside those collection intervals. Durations are
diagnostic observations, not a benchmark or a forecast of production GC cost.

## Resource and identity checks

AMD EPYC 9V74, logical CPU 2, normal .NET 10.0.8 settings and exact product
`1d10d22` binaries. The only deliberate intervention is the consumer's three
explicit collections, with added memory telemetry outside public call timing.
No LOKAD/DOTNET/COMPlus overrides are present. No native ORT is loaded by the
managed consumer. Model, input, tokenizer and consumer identities are frozen.

All {resource['samples']:,} process-group resource samples pass the original
3,600-second, 14 GiB RSS, 1 GiB available-memory and 32 MiB free-disk guards.
Peak sampled group RSS is **{resource['peak_rss']:,} bytes**;
minimum available memory is **{resource['min_available']:,} bytes**. Every sampled
worker thread remains on CPU 2. All original supervisor and worker births are
absent before collection and again before reporting.

The complete installed .NET 10.0.8 runtime and host are pinned ({len(frozen['managed_runtime']['files'])}
files), along with the inherited model/environment inventory. Frozen manifest
SHA-256: `{pin(base/'frozen.json')['sha256']}`. The collection verifies
{receipt['external_verified']:,} external files and binds {len(receipt['files'])} private/output files.
All raw requests, before/after memory snapshots, collector state, resource samples
and three collection records remain under
`artifacts/whisper-memory-collection-20260920`.

## Interpretation and limits

These observations test reclaimability in one process with a fixed intervention.
They do not identify every retained object's owner, prove the absence of a leak
at arbitrary request counts, or qualify the original normal-runtime workload.
`GC.GetGCMemoryInfo` heap, fragmentation, commitment and memory-load fields describe
the last completed collection; they must not be mistaken for current live samples.
The complete snapshots, including reported memory limits and collector indices,
are retained in [the observations](observations-20260920.json).

Fourteen damaged real application, collection and resource records are rejected.
Two protocol tests additionally exercise eleven corruptions and explicitly accept
an intervention that fails to reduce memory. The audit does not require its
reclaimability hypothesis to succeed. Existing encoder/logit numerical failures,
historical Windows timing and the completed AMD Parakeet/pyannote comparison are
unchanged. A reduction in fresh allocations through bounded buffer reuse is a
separate possible implementation hypothesis; it requires its own complete
correctness and normal-runtime resource evaluation. No forced GC is added to
production by this diagnostic.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    benchmark=ROOT/'BENCHMARK.md';s=benchmark.read_text(encoding='utf-8')
    marker='below. The later two-family timing scope preserves that resource failure.\n'
    assert s.count(marker)==1
    s=s.replace(marker,marker+'A separate [collection diagnostic](tests/whisper/memory-collection/results-20260920.md)\ncompletes the twenty requests with explicit collections after requests 8, 16 and\n20. It measures reclaimability and does not supply normal-runtime timing.\n')
    benchmark.write_text(s,encoding='utf-8')
    support=ROOT/'docs/model-support.md';s=support.read_text(encoding='utf-8')
    s+='\nThe subsequent [Whisper collection diagnostic](../tests/whisper/memory-collection/results-20260920.md)\ncompletes the same twenty short requests using explicit collections after requests\n8, 16 and 20, with unchanged results and ownership checks. It records managed-heap\nand RSS changes separately; it does not qualify normal repeated operation or add\nforced GC to production. The original AMD failure remains unchanged.\n'
    support.write_text(s,encoding='utf-8')
    snapshots=BASE/'closure-snapshots';snapshots.mkdir()
    for path in [benchmark,support,*sorted(folder.glob('*.py'))]:
        target=snapshots/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data]})
    write(BASE/'closed.json',dict(passed=True,normal_runtime_qualification=False,benchmark=False,
        closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,births=receipt['births']))
    print(json.dumps(dict(passed=True,closed=pin(BASE/'closed.json'),files=len(files))))


if __name__=='__main__':main()
