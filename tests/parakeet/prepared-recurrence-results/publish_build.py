"""Publish the compiled scope and preserved refusal without copying raw inventories."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-build-review-20260924'
BUILD=ROOT/'artifacts/parakeet-prepared-recurrence-build-amd-20260924'
OUT=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    assert pin(BASE/'closed.json')['sha256']=='1b7f8e2130d490851e861790b2e051edfca3308be6d6d9e1067cda8b97f9869f'
    proof=read(BASE/'closed.json');assert proof['passed'] and proof['paths_relative_to_repository']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    refusal=read(BUILD/'closed.json');assert not refusal['passed'] and refusal['build_jobs_passed']
    for name,wanted in refusal['files'].items():assert pin(BUILD/name)==wanted,name
    analysis=read(BASE/'analysis.json')
    observation=dict(passed=True,closure=pin(BASE/'closed.json'),analysis=analysis,renames=read(BASE/'renames.json'),
        terminal=proof['remote_terminal'],publisher=pin(Path(__file__)))
    with (OUT/'build-20260924.json').open('x',encoding='utf8') as stream:json.dump(observation,stream,indent=2);stream.write('\n')
    text=f'''# Bounded Parakeet recurrent preparation: isolated build

The source candidate compiled on AMD with SDK10.0.204/runtime10.0.8. The compiled
review accepts its declared scope: **10 changed existing methods, 61 new internal
methods, 3,179 unchanged Core methods and all 697 unchanged Data methods**.
Public interfaces and every existing implementation flag remain exact. The
existing ordered projection and panel kernels are byte-identical. No numerical
or performance result is claimed by this build.

Candidate Core: `{analysis['built']['Lokad.Onnx.dll']['sha256']}`.

Candidate Data: `{analysis['built']['Lokad.Onnx.Data.dll']['sha256']}`.

Source receipt: `{analysis['source_prepared']['sha256']}`.

The isolated implementation prepares W/R transposes once, owns them separately
from graph tensor bindings, and admits complete pairs only within the existing
aggregate budget. Graph invalidation, refresh accounting, execution contexts and
provider options carry the new internal map. Dispatch requires forward one-step,
batch-one, input/hidden-size-640 geometry and hardware SIMD. The arithmetic kernel,
gate epilogue and scalar/direct/unsupported fallbacks retain their original code.
The encoder/decoder caps remain 256/64 MiB. Root product remains selected.

The first scope wrapper stopped after compilation because adding a graph field
renumbered compiler-generated identifiers. That refusal is retained as
`b2d46d3b37e0df3d2df3828ab186760574feaaeb221c53f2346f8272dd5fdd6f`.
The [separate review](../prepared-recurrence-build-review/README.md) proves
28 renamed bodies and seven additional callers differ only in explicit compiler
identifiers. It then applies the original scope checker unchanged. Four mutation
tests reject instruction, flag, public-interface and unrelated-helper changes.
A local reviewer setup error expecting an absent log field is documented there.
**No product was rebuilt or rerun** to perform this correction.

All four build jobs exited zero. All 75 resource samples pass, with peak owned
RSS **511,041,536 bytes**. Every owner is terminal. Raw DLLs, complete inventories,
logs and source inputs remain in
`artifacts/parakeet-prepared-recurrence-build-amd-20260924`;
the independent review is
`artifacts/parakeet-prepared-recurrence-build-review-20260924`.
Review closure SHA-256: `{pin(BASE/'closed.json')['sha256']}`.

[Full method/rename/resource observations](build-20260924.json) provide the exact
scope. [Source and focused test design](../prepared-recurrence-source/README.md)
describe the remaining lifecycle/dispatch checks. Full native/public numerics,
actual model residency, fixed complete-call/application comparisons and release
regressions remain required before promotion or BENCHMARK.md changes.
'''
    with (OUT/'build-20260924.md').open('x',encoding='utf8') as stream:stream.write(text)
    print(json.dumps(dict(report=pin(OUT/'build-20260924.md'),observation=pin(OUT/'build-20260924.json'))))


if __name__=='__main__':main()
