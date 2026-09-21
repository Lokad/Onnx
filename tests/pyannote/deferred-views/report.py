"""Publish closed build and graph qualification without a latency claim."""
import shutil
from common import *


def main():
    assert not (TOOLS / 'results-20260922.md').exists()
    artifacts = {
        'focused': (BASE, 'focused-closed.json', 'focused-analysis.json'),
        'graphs': (BASE, 'closed.json', 'analysis.json'),
        'shared': (ROOT / 'artifacts/pyannote-deferred-views-shared-20260922', 'closed.json', 'analysis.json'),
        'parakeet': (ROOT / 'artifacts/pyannote-deferred-views-parakeet-20260922', 'closed.json', 'analysis.json')}
    proofs, analyses, identities = {}, {}, {}
    for name, (folder, closed, analysis) in artifacts.items():
        proof = read(folder / closed)
        assert proof['passed'] and proof['analysis'] == pin(folder / analysis)
        verify(proof['files'])
        for identity in proof.get('identities', proof.get('terminal_identities', [])):
            terminal(identity)
            identities[(identity['pid'], identity['birth'])] = identity
        proofs[name] = pin(folder / closed)
        analyses[name] = read(folder / analysis)
    old_base = ROOT / 'artifacts/pyannote-convolution-portable-rows-20260921'
    assert pin(old_base / 'closed.json')['sha256'] == '27ff741eb0a6aceb354806a030730175a95e429d24ae5c307a31a7633b756669'
    old_proof = read(old_base / 'closed.json')
    verify(old_proof['files'])
    old = read(old_base / 'analysis.json')
    allocations = []
    for order in ['forward', 'reverse']:
        def allocation(value):
            return next(r['allocated_mean'] for r in value['summaries']
                if (r['order'], r['graph'], r['mode'], r['phase']) == (order, 'embedding', 'reuse', 'repeat'))
        before, after = allocation(old), allocation(analyses['graphs'])
        allocations.append(dict(order=order, retained_predecessor_bytes=before, candidate_bytes=after, ratio=after / before))
    samples = sum(analyses[k]['resource_samples'] for k in ['focused', 'graphs'])
    samples += sum(r['samples'] for r in analyses['shared']['resources']) + analyses['parakeet']['samples']
    details = dict(closures=proofs, core=analyses['focused']['core'], data=analyses['focused']['data'],
        suites=analyses['focused']['suites'], graph_summaries=analyses['graphs']['summaries'],
        graph_calls=analyses['graphs']['calls'], graph_values=analyses['graphs']['values'],
        shared=analyses['shared'], parakeet=analyses['parakeet'], allocations=allocations,
        resource_samples=samples, terminal_identities=list(identities.values()),
        scope='Source/test/graph regression qualification only; full public and fresh timing remain separate.')
    save(TOOLS / 'observations-20260922.json', details)
    shutil.copy2(BASE / 'candidate.patch', TOOLS / 'candidate.patch')
    table = '\n'.join(f"| {r['order']} | {r['retained_predecessor_bytes']:,.0f} | {r['candidate_bytes']:,.0f} | {r['ratio']:.4f} |" for r in allocations)
    closures = '\n'.join(f"- {name}: `{value['sha256']}` ({value['bytes']:,} bytes)." for name, value in proofs.items())
    (TOOLS / 'results-20260922.md').write_text(f'''# Deferred tensor views: source and captured-model qualification

The one-method candidate passes the normal build, **214 focused cases**, **50
hardware-disabled cases**, **3,290 backend tests with 93 skips**, and **342
tensor tests**. All **108 captured graph calls / 17,502,642 output values**
preserve the qualified predecessor exactly, including input and held-output
ownership. Full public requests and matched ORT timing follow separately.

The [patch](candidate.patch) moves three DenseTensor wrapper constructions into
the fallback branch of RunTiledBatchFloat. The specialized helper receives the
same three sliced spans. Its predicate, matrix arithmetic, packing, clearing,
bias order and scratch ownership remain unchanged. Public convolution already
validates geometry and converts inputs to array-backed DenseTensor storage;
span slicing and the specialized helper retain their bounds checks.

The instruction/public-declaration checker permits exactly that one Core
method difference: **3,107 other Core methods and all 697 Data methods match**,
with no added or removed methods and identical checked public declarations.
Ordinary project references rebuild the candidate; Data's digest changes with
the build, despite unchanged instructions. This is not the older measured
Core5c0/Datae9 or the queued AMD payload.

## Measured graph allocations

| Process order | Retained predecessor bytes | Candidate bytes | Ratio |
|---|---:|---:|---:|
{table}

These are mean cumulative allocated bytes for repeated embedding graph calls
with reused contexts. The predecessor is the retained portable-row graph run,
Core5c0/Data1d; both use the same captured graph inputs and normal .NET10.0.12.
Current exact-predecessor whole-application allocation and timing will be
measured separately. Counters include process/runtime allocation; their
variation is retained. They are not resident memory or a latency speedup.
Every fresh/reused/disabled-cache case and all counters remain in the raw
closed evidence and [observations](observations-20260922.json).

## Affected-model regression

All **166 shared-model arrays / 5,000,814 values** pass native bounds and
remain unchanged. The Parakeet trajectory preserves **784 arrays / 3,090,494
values** exactly. Its three existing native discrepancies remain at
0.0002321004867553711 on the English duration output; this experiment does
not include the separate qualified Parakeet arithmetic fix and does not claim
that its native gate passes. All original public decisions still pass.

All **{samples} resource samples** and **{len(identities)} distinct process identities**
pass their independent audits. Builds/tests/captured/shared phases retain
900-second bounds and Parakeet1800seconds, with8GiBRSS,1GiBminimum available,
20GiBdisk and1GiBoutput. Preflight requires8GiB for builds and10GiB for inference.
The [premature Parakeet launch refusal](../deferred-views-qualification/launch-order-refusal-20260922.md)
is preserved; no inference began until preparation actually exited successfully.

## Reproduction and evidence

Use C:/Python313/python.exe -X utf8 -B with prepare.py, audit_preparation.py,
probe.py and audit_probe.py in this directory, reaping each actual process
before dependent work. Shared and Parakeet prepare/run/audit stages are in
../deferred-views-qualification. Existing outputs are refused. No root product,
VM owner, primary AMD payload or accepted latency table changes in this phase.

Core: `{details['core']['sha256']}`.
Data: `{details['data']['sha256']}`.

{closures}
''', encoding='utf8')
    print(json.dumps(dict(samples=samples, identities=len(identities), allocations=allocations, report=str(TOOLS / 'results-20260922.md'))))


if __name__ == '__main__':
    main()
