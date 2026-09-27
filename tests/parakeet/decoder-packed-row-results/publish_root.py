"""Publish the qualified actual root and package, without assigning new clocks."""
import json
from pathlib import Path
import sys
from publish_application import pin, read

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
sys.path.insert(0, str(ROOT / 'tests/parakeet/decoder-packed-row-root-amd'))
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
    assert len(applied['source_files']) == 441 and len(applied['changed']) == 3
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
    prose = f'''# Prepared single-row weights: actual root and package qualification

**All root and package checks pass.** The 441 build inputs match the measured
prepared-row product plus its portable four-fact integration fixture.
Two prepared-MatMul dispatch methods change, with one internal row kernel added.
The kernel reads the existing packed weights while preserving arithmetic,
reduction order and vector width. The real Parakeet fixture and original
process/product guards remain in the closed experimental contract evidence.
Root qualification checks actual runtime, product, consumer and CPU identities
independently.

A normal SDK 10.0.204 build preserves all 3,284 Core and 697 Data method bodies,
resolved operands, locals, exception regions and implementation flags relative
to the measured candidate. Public declarations and assembly attributes also
match exactly. There is no metadata exception or compiler-flag change.

| Full suite | Normal pass / skip | AVX512 disabled pass / skip |
|---|---:|---:|
| Backend | 3,564 / 42 | 3,474 / 132 |
| Tensor | 394 / 0 | 394 / 0 |

Every prior test name and outcome remains present, with four additional passing
prepared-row facts in each hardware mode. These cover 44 synthetic public calls
and 80 guarded raw cases: dimensions, batches, explicit execution modes, missing
and replaced weights, tails, exceptional values and NaN payloads, zero raw
allocation, immutable inputs and independently owned outputs. Both repository
source-policy tests remain intact. The original normal, AVX512-disabled and
hardware-disabled [contracts](contracts-20260927.json) retain all 270 public
comparisons and 162 raw comparisons, including the real model fixture.

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
The isolated prepared-row screen remains rejected: two repeatability controls
and two fallback cases failed. The first-call-only explanation was also rejected;
cache competition remains an inference. Accepting the Parakeet gain does not
establish universal fallback performance equivalence; these limitations remain
in the [screen](screen-20260927.md) and
[diagnostic](unmapped-calls-20260927.md) reports.

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
