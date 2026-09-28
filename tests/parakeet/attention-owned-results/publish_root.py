"""Publish the qualified actual root and package, without assigning new clocks."""
import json
from pathlib import Path
import sys
from common import pin, read, application

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-attention-owned-root-recovery-amd-20260928'
sys.path.insert(0, str(ROOT / 'tests/parakeet/attention-owned-root-recovery-amd'))
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
    assert len(applied['source_files']) == 446 and len(applied['changed']) == 2
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
    prose = f'''# Owned attention weights: actual root and package qualification

**All root and package checks pass.** The 446 build inputs preserve the measured
attention preparation product. The portable fixture uses explicit arguments in
place of identical defaults to comply with repository source policy.
The single policy change prepares 92 additional constant square attention weights
once, using the existing owned representation. Arithmetic leaves, reduction
order and the 256 MiB clone-cache limit remain unchanged.
Root qualification checks actual runtime, product, consumer and CPU identities
independently.

A normal SDK 10.0.204 build preserves all 3,288 Core and 697 Data method bodies,
resolved operands, locals, exception regions and implementation flags relative
to the measured candidate. Public declarations and assembly attributes also
match exactly. There is no metadata exception or compiler-flag change.

| Full suite | Normal pass / skip | AVX512 disabled pass / skip |
|---|---:|---:|
| Backend | 3,597 / 43 | 3,507 / 133 |
| Tensor | 394 / 0 | 394 / 0 |

Every prior test name and outcome remains present. The fixture adds 29 passing
attention preparation cases in each hardware mode and one expected skip for its
separate disabled-FMA contract. That case passed in the focused intrinsics-
disabled process. AVX512-disabled full tests still have FMA available.

The preparation cases check exact eligibility, cache limits, logical bits,
original input ownership, aliases, shared/visible weights, repeated preparation,
all corpus geometries and unsupported execution fallbacks. Both repository
source-policy tests remain intact. Focused qualification passed all 212 cases;
the exact-model census confirmed 179 owned weights and unchanged logical values,
37 cache entries and the 256 MiB cap in both hardware modes. See the
[focused contracts](contracts-20260928.md), [census](census-20260928.md) and
[release protocol](../attention-owned-root-recovery-amd/README.md).

The first root run failed the existing source-policy test because the new helper
declared five optional parameters. That failed collection remains preserved at
`9c757ee0`; no successful closure was assigned to it. This fresh qualification
uses only the explicit-argument fixture repair. All product inputs and completed
performance comparisons remain unchanged.

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
[Parakeet](application-20260928.md), [Pyannote](pyannote-application-20260928.md)
and [eight graph cases](graphs-20260928.md) bind the measured candidate above.
The application comparison admits a 1.128490% matched gain. The candidate takes
44.97562548 seconds versus ORT 39.2002844835, ratio 1.147329058. This is the net
measured effect of one preparation-policy change; the <=1.05 parity target
remains open. These package checks assign no additional timing benefit.

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
