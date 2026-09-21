"""Report a completely audited private sharing run; never claim matched latency."""
from pathlib import Path
import datetime,json,shutil
from deploy import BASE,ROOT,PRODUCT,ssh
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
    local=read(PRODUCT/'local-closed.json');assert local['passed'] and read(PRODUCT/'local-final-verification.json')['passed']
    for name,wanted in local['files'].items():assert pin(ROOT/name)==wanted,name
    terminal(receipt['births']);folder=Path(__file__).parent;gate=value['gate']
    report=folder/'results-20260920.md';data=folder/'observations-20260920.json';write(data,value)
    rows=[]
    for observation in value['observations']:
        resource=observation['resource'];sharing=observation['sharing']
        rows.append(f"| {'Conformance' if observation['phase']=='conformance' else 'Endurance'} | {observation['calls']} | {resource['peak_rss']:,} | {resource['min_available']:,} | {sharing['shared_payload_bytes']:,} |")
    text=f'''# Corrected Whisper decoder-weight sharing qualification on AMD

The private prototype completes **20 conformance and 80 endurance requests**
under normal .NET 10.0.8 on AMD EPYC 9V74, logical CPU 2. All exact transcript,
token, stop/no-speech, input, held-output and repeated-result checks pass.
Both processes verify **635,187,200 logical shared initializer bytes** and
independently confirm the backing arrays are shared between the two decoders.
Complete initializer bytes, types and shapes, graph bindings, packed-weight
totals and unique-storage counts remain unchanged after inference. Exactly two
prepared transpose tensor names acquire their verified node-output names, matching
the independently tested shared/unshared controls on both hosts.

| Worker | Requests | Peak sampled RSS bytes | Minimum available bytes | Actual shared initializer payload bytes |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

These are finite resource and application results, **not matched Microsoft ORT
timing**, a before/after RSS comparison or production integration. Raw elapsed
times remain diagnostic. The existing encoder/logit numerical failures remain.
The [local model-storage proof](../weight-sharing/local-results-20260920.md) independently measures
635,187,200 fewer unique initializer bytes; it does not predict equal RSS savings.

## Allocation and ownership

The candidate inherits the [qualified buffer reuse](../buffer-reuse/results-20260920.md)
with 512 MiB encoder and 128 MiB per decoder released-array caches, each capped
at 256 arrays. Every warm encoder call allocates at most
{gate['encoder_warm_max']:,} fresh pool bytes, within the fixed 16 MiB limit.
For matching conformance calls 2–16, public allocation is
**{gate['prototype_allocated_bytes']:,} bytes**, versus the original saved
**{gate['original_allocated_bytes']:,} bytes** ({gate['ratio']:.3%}).
The allocation improvement primarily belongs to inherited buffer reuse;
this experiment does not attribute it to weight sharing.

Sharing is private to one transcriber. Only complete contiguous FP32 arrays of
at least 4 KiB, equal shape and exactly equal bytes are eligible. Hashes nominate
candidates; full byte comparison establishes equality. Wrappers and names stay
separate, observable graph inputs/outputs are excluded, and the destination graph
rebuilds its own prepared clones. Neither global state nor arithmetic changes.

The consumer reads hashes and reference identities outside the public request
timers and allocation counters. It retains actual prior output objects and PCM
throughout each process. Pool counters cover released arrays; they do not bound
live inputs, attention caches, output objects or all managed memory. Decoder
pool counters describe the last execution of each public request only.

## Protocol and evidence

The second process starts only after the first process and all its prospective
application, storage, allocation and resource checks pass. Both keep the original
3,600-second, 14 GiB RSS, 1 GiB available-memory and 32 MiB free-disk guards, with
13 GiB memory/64 MiB disk preflight. All sampled threads use CPU 2. No explicit
garbage collection or LOKAD/DOTNET/COMPlus override is used. Process accounting
retains the limitations of before/after snapshots for short-lived processes.

The private product previously passed 3,101 backend tests with 93 skips, including
twelve new exact-byte, ownership and prepared-graph cases. The diagnostic consumer
build passes with zero warnings/errors. Each independent C# and Python transition checker accepts four actual controlled
transitions and rejects twenty-eight damaged snapshots. The combined Python
sharing/schema check also rejects six damaged records; the completed-run audit rejects {value['refusals']} damaged
request, resource, allocation and storage records. All original process births
are terminal before collection and again before this report.

Frozen manifest SHA-256: `{value['frozen']['sha256']}`. Collection retains
{len(receipt['files'])} files and verifies {receipt['external_verified']:,} external
runtime/model/reference identities. [Complete observations](observations-20260920.json)
retain every public request and resource summary. Full hashes, source, tests,
binaries, raw samples and prior local proof are under
`artifacts/whisper-weight-sharing-v2-20260920`.

The [original sharing campaign](../weight-sharing/failure-20260920.md) remains
failed: its unconditional metadata equality asserted before saving the mismatching
snapshots. [Controlled decoder diagnosis](../weight-metadata/results-20260920.md)
then identified the two existing cached tensor-name changes. This new consumer
saves both snapshots before assertion and permits only those exact transitions;
it preserves every other field, public call and resource gate. This is a separate
qualified campaign, not retroactive acceptance of the earlier run.

Production integration, broader cancellation/recovery/concurrent speech and
recording qualification, and an AMD native timing comparison remain separate work.
This report does not replace any earlier failed memory or numerical observation.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    benchmark=ROOT/'BENCHMARK.md';s=benchmark.read_text(encoding='utf-8')
    marker='byte unchanged in both shared and unshared controls. A corrected consumer is\nprepared; its full application/endurance qualification remains pending.\n';assert s.count(marker)==1
    s=s.replace(marker,'byte unchanged in both shared and unshared controls. The separate\n[corrected sharing campaign](tests/whisper/weight-sharing-v2/results-20260920.md)\nnow passes all 100 requests and its full weight, allocation and resource checks.\nProduction integration and matched AMD Whisper timing remain pending.\n')
    benchmark.write_text(s,encoding='utf-8')
    support=ROOT/'docs/model-support.md';s=support.read_text(encoding='utf-8')
    s+='\nThe [private Whisper decoder-weight sharing candidate](../tests/whisper/weight-sharing-v2/results-20260920.md)\nalso passes 20 conformance and 80 endurance requests on AMD, including complete\ndecoder payload and explicit cached-name transition checks. Production integration, broader recovery\nand recording qualification, numerical gaps and matched AMD latency remain open.\n';support.write_text(s,encoding='utf-8')
    snapshots=BASE/'closure-snapshots';snapshots.mkdir()
    for path in [benchmark,support,*sorted(folder.glob('*.py'))]:
        target=snapshots/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data]})
    write(BASE/'closed.json',dict(passed=True,prototype_only=True,benchmark=False,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,births=receipt['births']))
    print(json.dumps(dict(passed=True,closed=pin(BASE/'closed.json'),files=len(files))))


if __name__=='__main__':main()
