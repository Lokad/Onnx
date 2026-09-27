"""Publish the qualified actual root and package, without assigning new clocks."""
import json
from pathlib import Path
import sys
from common import pin, read, application

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-pointwise-tail-root-amd-20260927'
sys.path.insert(0, str(ROOT / 'tests/parakeet/pointwise-tail-root-amd'))
from source_scope import verify_source, root_files, verify_root
from checks import inventory, suite, suite256, package
from warning_census import compare


def main():
    proof = read(BASE / 'closed.json')
    value = read(BASE / 'analysis.json')
    assert proof['passed'] and value['passed'] and proof['analysis'] == pin(BASE / 'analysis.json')
    for name, wanted in proof['files'].items():
        assert pin(BASE / name) == wanted, name
    applied = read(BASE / 'bundle/evidence/root-applied.json')
    assert applied['source_files'] == root_files(verify_source())
    assert len(applied['source_files']) == 445 and len(applied['changed']) == 3
    verify_root(applied['source_files'])
    assert value['root_source_verified'] and value['root_integration'] == pin(BASE / 'collected/evidence/root-applied.json')
    assert value['measured'] == application()['identities']['candidate']
    assert value['inventory'] == inventory(read(BASE / 'collected/inventory/instructions.json'), value['measured'], value['built'])
    assert value['warnings'] == compare(BASE / 'collected')
    assert value['consumer']['passed'] and value['consumer']['ownership']
    for name, wanted in value['built'].items():
        assert pin(BASE / 'collected/runtime' / name) == wanted
    assert value['package'] == package(BASE / 'collected/nuget/Lokad.Onnx.0.2.0.nupkg', value['built']['Lokad.Onnx.dll'])
    for name in ['backend', 'tensors']:
        assert value['suites'][name] == suite(BASE / 'collected' / (name + '-tests') / (name + '.trx'), name, BASE / 'collected/evidence')
        assert value['suite256'][name] == suite256(BASE / 'collected' / (name + '-tests-256') / (name + '.trx'), name, BASE / 'collected/evidence')
    state = read(BASE / 'collected/identity.json')
    receipt = read(BASE / 'collected/collection.json')
    assert state['complete'] and state['code'] == 0 and receipt['terminal'] and receipt['code'] == 0
    assert len(state['runs']) == 18 and all(r['complete'] and r['code'] == 0 for r in state['runs'])
    paths = [OUT / 'root-20260927.md', OUT / 'root-observations-20260927.json']
    assert not any(p.exists() for p in paths), 'Preserve an existing publication'
    prose = f'''# Pointwise remainder sharing: actual root and package qualification

**All root and package checks pass.** The 445 build inputs match the measured
pointwise remainder product plus its portable two-fact integration fixture.
The packed matrix kernel shares the loaded vectors in its partial-column loops
across eight output rows. Packing, reduction order, complete 32-column panels
and the existing FMA versus multiply/add policy remain unchanged.
Root qualification checks actual runtime, product, consumer and CPU identities
independently.

A normal SDK 10.0.204 build preserves all 3,288 Core and 697 Data method bodies,
resolved operands, locals, exception regions and implementation flags relative
to the measured candidate. Public declarations and assembly attributes also
match exactly. There is no metadata exception or compiler-flag change.

| Full suite | Normal pass / skip | AVX512 disabled pass / skip |
|---|---:|---:|
| Backend | 3,568 / 42 | 3,478 / 132 |
| Tensor | 394 / 0 | 394 / 0 |

Every prior test name and outcome remains present, with two additional passing
remainder facts in each hardware mode. Their 111 raw geometries include widths
0–31, residual row groups, reduction and panel boundaries, and widths 222/224/225.
An independent integer matrix reference checks three accumulations into nonzero
outputs, guarded offsets, unchanged inputs and packed bytes, and zero repeated-
call allocation. Both repository source-policy tests remain intact.

Separate arithmetic qualification retains all 10,392 raw cases and 24 scalar
cases, requiring exact non-NaN bits and matching NaN classification. The original
failed NaN-payload campaign remains failed: the subsequent identical-baseline
diagnosis demonstrates that arithmetic NaN payloads are unstable even without a
product change. All observed payload differences remain explicit. The corrected
contract changes neither model tolerances nor complete public-result checks.
Both hardware modes' generated helpers retain eight accumulators and no vector
spills. See the [numerical diagnosis](../decoder-lstm-layout-profile-results/pointwise-tail-nan-20260927.md)
and the source/numerical bindings in the [release protocol](../pointwise-tail-root-amd/README.md).

The two existing CS8604 warning sites remain; every warning and summary matches
the qualified parent after normalizing only the campaign directory. No new or
suppressed compiler warning. The NuGet package contains the exact built Core
and only Google.Protobuf 3.33.5. Independent PackageReference consumption passes
model import, matrix and convolution calls, prepared spatial and Winograd
execution, immutable inputs and independently owned outputs.

Measured Core: `{value['measured']['Lokad.Onnx.dll']['sha256']}`.
Measured Data: `{value['measured']['Lokad.Onnx.Data.dll']['sha256']}`.
Built Core: `{value['built']['Lokad.Onnx.dll']['sha256']}`.
Built Data: `{value['built']['Lokad.Onnx.Data.dll']['sha256']}`.

All 18 workers and their supervisor are terminal. {sum(r['samples'] for r in value['resources']):,}
resource observations pass; peak owned RSS is {max(r['peak_rss'] for r in value['resources']):,} bytes.
This is package qualification, not a new timing measurement. The admitted
[Parakeet](../decoder-lstm-layout-profile-results/pointwise-tail-app-20260927.md), [Pyannote](pyannote-application-20260927.md)
and [eight graph cases](graphs-20260927.md) bind the measured candidate above.
The isolated component screen remains rejected because 38/246 repeatability
controls failed. The separate runtime diagnosis finds JIT activity inside
measured calls, but does not rescore the original clocks or resolve all component
variation. The full Parakeet comparison establishes independent application
benefit for this unchanged candidate. See the [screen](../decoder-lstm-layout-profile-results/pointwise-tail-timing-20260927.md) and
[runtime diagnosis](../decoder-lstm-layout-profile-results/pointwise-tail-runtime-20260927.md).

[Full test census, compiled scope, package and resource evidence](root-observations-20260927.json).
Closure: `{pin(BASE / 'closed.json')['sha256']}`.
'''
    with paths[0].open('x', encoding='utf8') as stream:
        stream.write(prose)
    with paths[1].open('x', encoding='utf8') as stream:
        json.dump(dict(closure=pin(BASE / 'closed.json'), **value), stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=True, closure=pin(BASE / 'closed.json'), built=value['built'])))


if __name__ == '__main__':
    main()
