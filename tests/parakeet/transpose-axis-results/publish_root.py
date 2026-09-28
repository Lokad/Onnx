"""Publish the qualified actual root and package, without assigning new clocks."""
import json
from pathlib import Path
import sys
from common import pin, read, application

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-transpose-axis-root-amd-20260928'
sys.path.insert(0, str(ROOT / 'tests/parakeet/transpose-axis-root-amd'))
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
    assert len(applied['source_files']) == 447 and len(applied['changed']) == 2
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
    paths = [OUT / 'root-20260928.md', OUT / 'root-observations-20260928.json']
    assert not any(p.exists() for p in paths), 'Preserve an existing publication'
    prose = f'''# Tiled axis movement: actual root and package qualification

**All root and package checks pass.** The 447 build inputs preserve the measured
transpose candidate. Only TransposeInto changes, reusing the existing tile for
rank-three and rank-four axis movement. Arithmetic, weight preparation and cache
limits remain unchanged. Runtime, product, consumer and CPU identities are checked.

A normal SDK 10.0.204 build preserves all 3,288 Core and 697 Data method bodies,
resolved operands, locals, exception regions and implementation flags relative
to the measured candidate. Public declarations and assembly attributes match.

| Full suite | Normal pass / skip | AVX512 disabled pass / skip |
|---|---:|---:|
| Backend | 3,603 / 43 | 3,513 / 133 |
| Tensor | 394 / 0 | 394 / 0 |

Every prior test name and outcome remains present. Six portable transpose facts
pass in each full mode. Focused qualification passed 487 checks in four modes,
including intrinsics-disabled and vector-faces-disabled processes. It covers
all 76 actual geometries, exceptional float bits, partial and empty faces,
reversed/sliced storage, alias guards and independently owned outputs.
The existing source-policy tests remain unchanged. See the
[focused contracts](contracts-20260928.md) and
[release protocol](../transpose-axis-root-amd/README.md).

The two existing CS8604 warning sites remain; every warning and summary matches
the qualified parent after normalizing only the campaign directory. No new or
suppressed compiler warning. NuGet contains the exact built Core and only
Google.Protobuf 3.33.5. Independent PackageReference consumption passes model
import, matrix and convolution calls, prepared spatial and Winograd execution,
immutable inputs and independently owned outputs.

Measured Core: `{value['measured']['Lokad.Onnx.dll']['sha256']}`.
Measured Data: `{value['measured']['Lokad.Onnx.Data.dll']['sha256']}`.
Built Core: `{value['built']['Lokad.Onnx.dll']['sha256']}`.
Built Data: `{value['built']['Lokad.Onnx.Data.dll']['sha256']}`.

All 18 workers and their supervisor are terminal. {sum(r['samples'] for r in value['resources']):,}
resource observations pass; peak owned RSS is {max(r['peak_rss'] for r in value['resources']):,} bytes.
These are package checks, with no new timing measurement. The admitted
[Parakeet](application-20260928.md), [Pyannote](pyannote-application-20260928.md)
and [eight graph cases](graphs-20260928.md) bind the measured candidate above.
Parakeet takes 43.459338592 seconds versus matched current 44.537545594 and ORT
39.202810428: a 2.420895% gain and ratio 1.108577118. The prospective saving of
at least 0.45 seconds is met. Parity <=1.05 remains open. Fresh attribution is
required before assigning the whole net gain to the diagnosed encoder nodes
or selecting a further optimization.

[Full test census, compiled scope, package and resource evidence](root-observations-20260928.json).
Closure: `{pin(BASE / 'closed.json')['sha256']}`.
'''
    with paths[0].open('x', encoding='utf8') as stream:
        stream.write(prose)
    with paths[1].open('x', encoding='utf8') as stream:
        json.dump(dict(closure=pin(BASE / 'closed.json'), **value), stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=True, closure=pin(BASE / 'closed.json'), built=value['built'])))


if __name__ == '__main__':
    main()
