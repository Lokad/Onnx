"""Close the successful corrected qualification while retaining all failed attempts."""
import datetime,shutil
from common import *


def main():
    base=ROOT/'artifacts/whisper-memory-product-v2-20260921';value=read(base/'audit.json');assert value['passed'] and value['prototype_only']
    assert all(absent(b) for b in value['births']) and value['methods_compared']==3777
    assert value['tensor_tests']['passed']=='342' and value['backend_tests']['passed']=='3101'
    folder=Path(__file__).parent;report=folder/'results-20260921.md';data=folder/'observations-20260921.json'
    write(data,value)
    rows=value['budgets'][0]['rows'];assert all(v['rows']==rows for v in value['budgets'])
    table='\n'.join(f"| {r['budget']} | {r['cold']} | {r['warm']} | Yes |" for r in rows)
    samples=sum(r['samples'] for r in value['resources']);peak=max(r['peak_sampled_group_rss'] for r in value['resources'])
    text=f'''# Corrected Whisper memory candidate: coherent build and private package

The corrected private candidate builds all eight solution projects with **zero
warnings and errors**, passes **3,101 backend tests** with 93 hardware skips and
**342 tensor tests**, and passes all four independent package-consumer settings.
The core package is restored from a new private feed/cache; its built, packaged,
cached and actually loaded DLL bytes match. No package is published.

The complete source comprises 488 archived files and the same five candidate
changes previously exercised by the audio qualification. A nullable flow annotation
replaces four null-forgiving operators in the private sharing helper; explicit test
overloads and typed placeholders satisfy repository source policies. No arithmetic
kernel or production default is changed.

All **3,777 Core/Data method bodies** compare equal with the earlier audio-tested
binaries after resolving metadata operands: 3,086 Core and 691 Data methods,
including local types, stack limits, branch operands and exception clauses. The
corrected helper has the intended `NotNullWhen(true)` annotation. The DLL hashes
differ, so this is an executable-method/source bridge, not binary identity or a new
model inference result. The complete normalized instruction records are retained.

## Actual package consumption

Four fresh processes cover default, fingerprint-only, wider-LayerNorm-only and
both diagnostic settings. Every process checks Relu, MNIST imported from file,
bytes and sliced memory, and all three normalization APIs. All **40,728** saved
normalization values pass independent scalar reconstruction at the unchanged
1e-6 limit. All outputs match across settings and no native ORT module is loaded.

The new public `CreateExecution(options, maximumReleasedBufferBytes)` overload is
called through the restored package. The table reports newly allocated pool bytes
for the same three-Add-node graph, before and after Reset; all four settings agree.

| Released-buffer budget, bytes | First call, bytes | Second call, bytes | Exact outputs and ownership |
|---|---:|---:|---|
{table}

Every call returns four times its input. Both original inputs and actual retained
outputs remain exact after the next call. Each process also rejects a negative
budget. These are bounded cache/ownership checks, not process-memory limits.

All **{samples:,} resource samples** pass the prospective per-stage 300-second,
4 GiB process-group RSS and 1 GiB available-memory limits, with CPU0 affinity.
Peak sampled process-group RSS is **{peak:,} bytes**. All recorded process
identities are terminal. Build and consumer environments leave global settings
and unrelated local work unchanged.

## Preserved failures and scope

The [first source qualification](../memory-product/failure-20260921.md) retains its
two failed repository-policy tests and four nullable warnings. The corrected
source then passes both full suites. Its first synthetic package graph omitted
the required `Name` metadata and failed before producing its complete result.
Only that consumer was corrected and rebuilt, reusing the successful product
build and package. All four corrected consumers pass.

The first method-inspection helper used a collectible load context whose lifetime
ended before dependency resolution; its exception and source remain saved. A
distinct helper with process-lifetime contexts completes all method comparisons.
No model inference, successful product build, test suite or package creation was
repeated to repair either inspection error.

This remains a private candidate. Separate AMD endurance and public-contract
evidence, coherent production integration and matched AMD Whisper/ORT timing
remain distinct requirements. Existing strict numerical failures are unchanged.
The core NuGet package excludes Data/CLI; audio APIs are repository-project APIs.

Package SHA-256: `{value['package']['sha256']}` ({value['package']['bytes']:,} bytes).
Core SHA-256: `{value['core']['sha256']}` ({value['core']['bytes']:,} bytes).
[Complete observations](observations-20260921.json) retain every stage/resource,
normalization, budget and identity summary. Full source, Git inventories, patches,
test results, original failures, package/cache selection and actual outputs are
under `artifacts/whisper-memory-product-v2-20260921`.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    snapshots=base/'closure-tools';snapshots.mkdir()
    for p in sorted(folder.iterdir()):
        if p.is_file():shutil.copyfile(p,snapshots/p.name)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data]})
    write(base/'closed.json',dict(passed=True,prototype_only=True,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,births=value['births']))
    print(json.dumps(dict(passed=True,closure=pin(base/'closed.json'),files=len(files),resource_samples=samples)))


if __name__=='__main__':main()
