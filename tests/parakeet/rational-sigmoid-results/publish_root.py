"""Publish the qualified actual root and package, without assigning new clocks."""
import json
from pathlib import Path
import sys
from publish_application import pin, read

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-rational-sigmoid-root-amd-20260927'
sys.path.insert(0, str(ROOT / 'tests/parakeet/rational-sigmoid-root-amd'))
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
    assert len(applied['source_files']) == 439 and len(applied['changed']) == 3
    verify_root(applied['source_files'])
    assert value['root_source_verified'] and value['root_integration'] == pin(BASE / 'collected/evidence/root-applied.json')
    assert value['measured'] == read(OUT / 'application-20260927.json')['identities']['candidate']
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
    prose = f'''# Rational sigmoid: actual root and package qualification

**All root and package checks pass.** The 439 build inputs match the measured
rational sigmoid product plus its portable eight-fact integration fixture.
Only public Sigmoid changes, with one private rational-vector helper added.
All eight arithmetic/ownership fact bodies remain unchanged; helper defaults
become two explicit forwarding overloads. The VM-only identity fact remains in
its already qualified experimental source and results. Root qualification checks
actual runtime, product, consumer and CPU identities independently.

A normal SDK 10.0.204 build preserves all 3,283 Core and 697 Data method bodies,
resolved operands, locals, exception regions and implementation flags relative
to the measured candidate. Public declarations and assembly attributes also
match exactly. There is no metadata exception or compiler-flag change.

| Full suite | Normal pass / skip | AVX512 disabled pass / skip |
|---|---:|---:|
| Backend | 3,560 / 42 | 3,470 / 132 |
| Tensor | 394 / 0 | 394 / 0 |

Every prior test name and outcome remains present, with eight additional passing
sigmoid facts in each hardware mode. These cover finite/bit-pattern sweeps,
rounding boundaries, special values, every vector tail, scalar/double behavior,
layouts, validation and independent output ownership. Both repository source-policy
tests remain intact. The separate intrinsics-disabled sigmoid qualification is
retained with its original tolerance and full sweep evidence.

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
[Parakeet](application-20260927.md), [Pyannote](pyannote-application-20260927.md)
and [eight graph cases](graphs-20260927.md) bind the measured candidate above.
The isolated sigmoid screen remains rejected: thirteen repeatability controls
and four fallback cases failed. Double fallback latency remains unresolved.
Accepting the Parakeet gain does not establish universal fallback speed parity;
these limitations remain in the [screen](screen-20260927.md) and
[diagnostic](fallback-diagnosis-20260927.md) reports.

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
