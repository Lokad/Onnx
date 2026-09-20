"""Report a fully audited successful prototype without making a timing or product claim."""
from pathlib import Path
import datetime,json,shutil
from deploy import BASE,ROOT,REMOTE,ssh
from protocol import pin,read,write


def terminal(births):
    script='''import sys,json
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(terminal=True,births=%r)))
'''%(births,births)
    return json.loads(ssh(script))


def main():
    value=read(BASE/'audit.json');assert value['passed'] and value['prototype_only'] and not value['benchmark']
    assert not (BASE/'closed.json').exists();base=BASE/'collected';receipt=read(base/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    terminal(receipt['births']);gate=value['gate'];folder=Path(__file__).resolve().parent
    report=folder/'results-20260920.md';data=folder/'observations-20260920.json';write(data,value)
    rows=[]
    for observation in value['observations']:
        resource=observation['resource']
        rows.append(f"| {'Conformance' if observation['phase']=='conformance' else 'Endurance'} | {observation['calls']} | {resource['peak_rss']:,} | {resource['min_available']:,} | {observation['allocated_min']:,}–{observation['allocated_max']:,} |")
    text=f'''# Whisper bounded buffer reuse on AMD

The private prototype completes twenty conformance requests and eighty requests
in a separate endurance process without explicit garbage collection or runtime
overrides. Every transcript, token sequence, stop/no-speech decision, repeated
result, held output and PCM-preservation check passes. This is a finite memory
and allocation experiment, not a matched Microsoft ORT timing result or a
production integration.

For matching requests 2–16, public-call allocation falls from
**{gate['original_allocated_bytes']:,} to {gate['prototype_allocated_bytes']:,} bytes**
({gate['ratio']:.3%} of the saved original prefix). Warm encoder fresh pool payload
is at most **{gate['encoder_warm_max']:,} bytes**, passing the predeclared 16 MiB
limit. The original separate reused-encoder observation allocated 367,680,000
bytes on each warm call. Allocation counts describe managed allocation traffic,
not retained memory or maximum process RSS.

| Worker | Requests | Peak sampled RSS bytes | Minimum available bytes | Public allocation bytes/request |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

All sampled worker threads run on logical CPU 2 of the AMD EPYC 9V74 VM under
.NET 10.0.8. The existing 3,600-second, 14 GiB RSS, 1 GiB available-memory and
32 MiB disk limits pass. The second process starts only after the complete first
process and its prospective allocation/application/resource gates pass. The
four-pass worker includes one full corpus warmup and three further passes;
its elapsed-time fields remain diagnostic and are not added to BENCHMARK's
ORT comparison tables. Process accounting is retained, with the limitations of
before/after process snapshots.

## Change and scope

The private git archive starts at `18e10e3`. A new explicit execution-context
budget controls retained released arrays; the existing factory remains unchanged.
Whisper reuses three contexts under its existing instance lock: a 512 MiB encoder
cache and two 128 MiB decoder caches, each capped at 256 arrays. Existing reset,
failure-finally handling and per-request attention state remain. No arithmetic,
weights, decoding policy, input corpus, reference or tolerance changes.

The cache counters describe already-released arrays only. Contexts separately
retain their last input bindings, and decoder counters report the final decoder
execution in each public request, not a sum over its generated tokens. Public
allocation and process memory are therefore measured independently. Successful
short-clip endurance does not bound arbitrary recording length, concurrency or
all cancellation/recovery behavior; broader product qualification remains needed.

The first local suite attempt omitted the CLI prerequisite and retains four
failures, 3,085 passes and 93 skips. Supplying the CLI from the same source and
building it yields 3,089 passes, zero failures and 93 skips, including six new
explicit-budget cases. A packaging-path correction then obtains dependencies
from the CLI output without rerunning tests. Consumer build has zero warnings
and errors. Three protocol tests cover eleven corruptions and exact allocation
boundaries. The real-evidence audit rejects {value['refusals']} damaged records.

The [original normal-runtime failure](../../audio/amd-comparison/resource-failure-20260920.md)
and [explicit-collection diagnostic](../memory-collection/results-20260920.md)
remain unchanged. Existing encoder/logit numerical differences also remain;
this experiment does not waive them or claim parity with ORT.

## Evidence

Frozen manifest SHA-256: `{value['frozen']['sha256']}`. Collection verifies
{receipt['external_verified']:,} external files and retains {len(receipt['files'])}
private/output files. All original process births are terminal before collection
and again before reporting. [Complete observations](observations-20260920.json)
retain every public call, cache counter, heap/GC snapshot and resource summary.
Raw source archives, patches, original failed tests, successful tests, binaries,
all process samples and full manifests are under
`artifacts/whisper-buffer-reuse-20260920`.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    benchmark=ROOT/'BENCHMARK.md';s=benchmark.read_text(encoding='utf-8')
    marker='20. It measures reclaimability and does not supply normal-runtime timing.\n'
    assert s.count(marker)==1
    s=s.replace(marker,marker+'A later [private buffer-reuse prototype](tests/whisper/buffer-reuse/results-20260920.md)\ncompletes 20 conformance and 80 endurance requests without forced collection.\nIt reduces allocations but does not supply a matched AMD Whisper timing result.\n')
    benchmark.write_text(s,encoding='utf-8')
    support=ROOT/'docs/model-support.md';s=support.read_text(encoding='utf-8')
    s+='\nThe [private Whisper buffer-reuse prototype](../tests/whisper/buffer-reuse/results-20260920.md)\ncompletes twenty conformance and eighty endurance requests on AMD without forced\ncollection. Allocation and resource gates pass for that fixed short-clip workload.\nIt is not a production source change, a broader numerical qualification or an ORT\nlatency comparison. Its report preserves the original normal-runtime failure.\n'
    support.write_text(s,encoding='utf-8')
    snapshots=BASE/'closure-snapshots';snapshots.mkdir()
    for path in [benchmark,support,ROOT/'docs/source-review.md',*sorted(folder.glob('*.py'))]:
        target=snapshots/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data,folder/'ort-source-20260920.json']})
    write(BASE/'closed.json',dict(passed=True,prototype_only=True,benchmark=False,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,births=receipt['births']))
    print(json.dumps(dict(passed=True,closed=pin(BASE/'closed.json'),files=len(files))))


if __name__=='__main__':main()
