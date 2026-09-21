"""Report only a completed, audited AMD public-contract replay."""
from pathlib import Path
import datetime,json,shutil
from common import ROOT,BASE,LOCAL,PRELUDE,pin,read,write,ssh


def main():
    value=read(BASE/'audit.json');receipt=read(BASE/'collected/collection.json')
    assert value['passed'] and value['private_prototype'] and not value['benchmark']
    assert value['application']['calls']==13 and value['application']['refusals']==16 and value['damaged_records_rejected']==16
    for name,wanted in receipt['files'].items():assert pin(BASE/'collected'/name)==wanted,name
    assert read(LOCAL/'local-final-verification.json')['passed']
    ssh(PRELUDE+'terminal(%r)\n'%receipt['births'])
    folder=Path(__file__).parent;report=folder/'results-20260921.md';data=folder/'observations-20260921.json'
    write(data,value)
    rows=[]
    for row in value['application']['recordings']:
        rows.append(f"| {row['name']+(' (repeat)' if row['repeat'] else '')} | {row['windows']} | {row['segments']} | {row['stop_reason']} | Yes |")
    resource=value['resources']
    text=f'''# Whisper memory candidate: AMD public request contracts

The private candidate passes **13 completed requests and 16 refusal or
cancellation checks** on AMD EPYC 9V74, .NET 10.0.8, logical CPU 2. This uses the
same consumer and product DLLs as the independently closed
[Windows replay](../memory-contracts-v2/local-results-20260921.md).

| Recording | Audio windows | Segments | Stop reason | Exact native decisions |
|---|---:|---:|---|---|
{chr(10).join(rows)}

The other requests cover empty input, ten-minute digital silence, two concurrent
silent recordings, a short regression, two overlapping speech requests on one
transcriber and a short recovery after invalid and canceled requests. Actual speech
request lifetimes overlap **{value['application']['overlap_seconds']:.6f} seconds**.
Both requests match native text, token, stop and no-speech decisions. Complete own
repeated recording results and short recovery remain exact, including confidence.
All original PCM and actual held outputs are checked after subsequent requests.
Recording timelines are independently reconstructed from the token policy.
Confidence differences remain diagnostics in the complete observations.

All **{value['original_initializer_checks']:,} original initializer comparisons**
pass, before and after the sequence. The two source-proven cached transpose names
are the only permitted metadata changes. Every weight byte, shape, type, graph
binding and storage count remains exact. Actual shared backing arrays contain
**{value['sharing']['shared_payload_bytes']:,} bytes**, matching the independent
logical sharing census.

All **{resource['samples']:,} resource samples** and
**{resource['thread_observations']:,} thread-affinity observations** pass. Peak
sampled process-group RSS is **{resource['peak_rss']:,} bytes**; minimum system
available memory is **{resource['min_available']:,} bytes**. The fixed limits are
1,800 seconds, 14 GiB RSS, 1 GiB available memory and 32 MiB disk, with 13 GiB
available memory and 64 MiB disk required before launch. The maximum sample gap
is {resource['max_gap']:.6f} seconds. All recorded process identities are terminal.
No forced collection or LOKAD/DOTNET/COMPlus runtime override is used.

The auditor rejects sixteen damaged application, weight, process identity,
affinity and resource records. The existing private product previously passed
3,101 backend tests with 93 hardware skips. Its actual DLLs and the portable
consumer's DLL match the files used in both host replays; no rebuild or native
reference inference is performed for this host transfer.

## Scope and evidence

This qualifies finite recording, concurrency and recovery behavior of a private
memory candidate. It does not establish arbitrary-duration speech behavior,
production integration or a matched Microsoft ORT latency result. Existing
encoder/logit numerical failures and all earlier resource failures remain valid.
The [separate sharing campaign](../weight-sharing-v2/results-20260920.md) supplies
the preceding twenty/eighty-request AMD conformance and endurance evidence.

Frozen receipt SHA-256: `{value['frozen']['sha256']}`. The collection retains
{len(receipt['files'])} files and verifies {receipt['external_verified']:,} external
runtime/model identities. [Complete observations](observations-20260921.json)
retain decision summaries, overlap, confidence differences, resource totals and
result identities. All raw results, PCM, snapshots, source, binaries and resource
samples are retained under `artifacts/whisper-memory-contracts-amd-20260921`.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    benchmark=ROOT/'BENCHMARK.md';s=benchmark.read_text(encoding='utf-8')
    marker='Production integration and matched AMD Whisper timing remain pending.\n';assert s.count(marker)==1
    s=s.replace(marker,'The same private candidate also passes recording, concurrent speech and recovery\ncontracts on [Windows](tests/whisper/memory-contracts-v2/local-results-20260921.md)\nand [AMD](tests/whisper/memory-contracts-amd/results-20260921.md).\n'+marker)
    benchmark.write_text(s,encoding='utf-8')
    support=ROOT/'docs/model-support.md';s=support.read_text(encoding='utf-8')
    s+='\nThe same private memory candidate also passes [AMD public contracts](../tests/whisper/memory-contracts-amd/results-20260921.md): thirteen requests and sixteen refusal/cancellation checks, including recording, overlapping speech and exact recovery. Production integration, numerical gaps and matched AMD Whisper latency remain pending.\n'
    support.write_text(s,encoding='utf-8')
    snapshots=BASE/'closure-snapshots';snapshots.mkdir()
    for path in [benchmark,support,*sorted(folder.glob('*.py'))]:
        target=snapshots/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data]})
    write(BASE/'closed.json',dict(passed=True,private_prototype=True,benchmark=False,files=files,births=receipt['births'],closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(passed=True,closure=pin(BASE/'closed.json'),pins=len(files))))


if __name__=='__main__':main()
