"""Report complete local private-candidate contracts and seal their evidence."""
from pathlib import Path
import json,shutil
from prepare import ROOT,BASE,PRODUCT,pin,read,write
from audit import terminal


def main():
    value=read(BASE/'local-audit.json');assert value['passed'] and value['private_prototype'] and not value['benchmark']
    terminal(value['births']);assert not (BASE/'local-closed.json').exists()
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(BASE/name)==wanted,name
    for name,wanted in prepared['inputs'].items():assert pin(ROOT/name)==wanted,name
    folder=Path(__file__).parent;data=folder/'local-observations-20260921.json';report=folder/'local-results-20260921.md';write(data,value)
    app=value['application'];resource=value['resources'];rows=[]
    for row in app['recordings']:
        rows.append(f"| {row['name']}{' (repeat)' if row['repeat'] else ''} | {row['windows']} | {row['segments']} | {row['stop_reason']} | Yes |")
    text=f'''# Whisper memory candidate: Windows public request contracts

The private memory candidate passes **13 completed requests and 16 refusal or
cancellation checks** on Windows, .NET 10.0.12, logical CPU 2. The two concurrent
non-silent requests use the same transcriber and their measured request lifetimes
overlap. Both retain exact native text, tokens, stop and no-speech decisions.
After invalid and canceled requests, a successful short request reproduces the
earlier complete managed result exactly, including confidence fields.

| Recording | Windows | Segments | Stop reason | Exact native decisions |
|---|---:|---:|---|---|
{chr(10).join(rows)}

The remaining requests cover empty input, ten-minute digital silence, two
concurrent silent recordings, the existing short regression, two overlapping
speech calls and one short recovery. The finite connected and shifted recordings
do not establish arbitrary-duration speech behavior. The original repeated
recording also retains its complete managed result exactly.

All actual prior output objects and PCM inputs remain held and unchanged. The
independent auditor reconstructs recording segments and seek decisions from the
tokens and compares every public decision with the pinned native references.
Native confidence differences remain reported diagnostics. Scheduling cancellation
after 50 ms proves the observed cancellation and recovery; it does not locate the
precise neural instruction at which the token became canceled.

## Weight and resource checks

The complete before/after decoder snapshots preserve every initializer payload,
shape, type, membership, graph binding, packing total and shared-storage count.
Only the two independently verified cached transpose tensor names acquire their
graph-output names. All **{value['original_initializer_checks']:,}** comparisons of
original serialized initializer hashes, shapes and types pass. The candidate
reports and independently checks **635,187,200 shared initializer bytes**.

All **{resource['samples']:,}** resource samples pass the 1,800-second, 14 GiB RSS,
1 GiB available-memory and 32 MiB disk limits. Peak sampled RSS is
**{resource['peak_rss']:,} bytes** and minimum available memory is
**{resource['min_available']:,} bytes**. All actual process identities are terminal.
No forced collection or LOKAD/DOTNET/COMPlus override is used. These observations
are not a before/after RSS comparison or matched Microsoft ORT latency.

The diagnostic build has zero warnings/errors; the underlying unchanged private
product previously passed 3,101 backend tests, including twelve weight-sharing
cases, with 93 hardware skips. Its DLLs match those actually exercised by the test
process. The completed-result audit rejects **{value['damaged_records_rejected']}**
damaged application and weight records.

## Scope and evidence

The first contract attempt was refused at its local memory/disk preflight before
any child or inference launched. Its unused frozen artifact is preserved under
`whisper-memory-contracts-20260920`. This separate corrected consumer saves both
snapshots before validation and uses the source-proven two-name policy. The
[earlier sharing campaign failure](../weight-sharing/failure-20260920.md) and
[controlled decoder diagnosis](../weight-metadata/results-20260920.md) remain
separate evidence.

This is local qualification of a private candidate. AMD contract coverage,
coherent production integration, package checks and matched AMD Whisper timing
remain pending. Existing encoder/logit numerical failures are unchanged.

Frozen runtime/consumer receipt SHA-256: `{value['frozen']['sha256']}`.
[Complete observations](local-observations-20260921.json) retain each recording
decision summary, concurrency overlap, resource totals and result identities.
Full source, inputs, native references, actual output objects' serialized results,
weight snapshots and resource samples are under
`artifacts/whisper-memory-contracts-v2-20260921`.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    snapshots=BASE/'local-tool-snapshots';snapshots.mkdir()
    for p in sorted(folder.iterdir()):
        if p.is_file():shutil.copyfile(p,snapshots/p.name)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data]})
    write(BASE/'local-closed.json',dict(passed=True,private_prototype=True,benchmark=False,files=files,births=value['births']))
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    print(json.dumps(dict(passed=True,closure=pin(BASE/'local-closed.json'),pins=len(files),requests=13,refusals=16)))


if __name__=='__main__':main()
