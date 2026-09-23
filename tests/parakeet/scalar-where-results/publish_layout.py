"""Publish the closed unchanged-product capture, including its audit correction."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-scalar-where-layout-amd-20260923'
OUT = Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def main():
    assert pin(BASE / 'closed.json')['sha256'] == 'bc0942f51e27e70f33ae9fdc994c345fe3c620159478ced1a40d5433777f66d8'
    closure = read(BASE / 'closed.json'); assert closure['passed']
    for name, wanted in closure['files'].items(): assert pin(BASE / name) == wanted, name
    a = read(BASE / 'analysis.json'); result = read(BASE / 'collected/capture/result.json')
    correction = read(BASE / 'audit-reconciliation.json')
    fixtures = []
    for f in result['fixtures']:
        mask = (BASE / 'collected/capture' / f['inputs'][0]['file']).read_bytes()
        assert all(x in (0, 1) for x in mask)
        fixtures.append(dict(request=f['request'], name=f['name'], true_count=sum(mask),
            mask_count=len(mask), inputs=f['inputs'], output=f['output']))
    value = dict(closure=pin(BASE / 'closed.json'), analysis=a, fixtures=fixtures,
                 correction=correction, all_observations=result['requests'])
    path = OUT / 'layout-observations-20260923.json'; assert not path.exists()
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    text = '''# Selected Parakeet Where capture

Two complete encoder calls on the selected release reconcile exactly: all
217,090 output values, unchanged inputs, and retained outputs after graph reset
and reuse. The capture is diagnostic; it provides no latency measurement.

The logger observes all 2,856 prepared graph nodes in order and records all
73 Where nodes per request. There are 144 float observations in the initial
dense/scalar scope and two Int64 shape selections outside it. All observed
inputs and outputs are exact DenseTensor classes with standard row-major layout.

| Family | Condition | True input | False input / output | Observations |
| --- | --- | --- | --- | ---: |
| Attention | [1,1,T,T] | scalar -10000 | [1,8,T,T] | 48 |
| Attention cleanup | [1,1,T,T] | scalar +0 | [1,8,T,T] | 48 |
| Convolution | [1,1,T] | scalar +0 | [1,1024,T] | 48 |
| Shape selection | [2] | Int64 [2] | Int64 [2] | 2 |

T is 74 for english-16k and 138 for jfk-48k-stereo. Eight fixed fixtures
exported 36 arrays (5,793,008 bytes, below the predeclared 32 MiB cap). All six
exported float masks are entirely false; their outputs equal their false inputs
bit for bit. These mask contents are proved for the exported layer-0 fixtures,
not asserted for every layer. NumPy independently selects the operand bits for
every fixture element. Native complete-encoder normalized maximum errors are
1.5273690223693848e-6 and 4.0102750062942505e-6; lengths are exact.

The original audit rejected raw ONNX constant names after import had deduplicated
identical scalars. The unchanged ConstFold.DedupeConstants implementation compares
dtype, shape and bits before replacing bindings. An independent read of the pinned
ONNX graph proves 48 +0 and 24 -10000 rank-zero Float constants and all 140 changed
bindings across both requests. Node names/order and the other operands are exact.
Six checker tests reject wrong condition/true/false bindings, node names and arity.
The separate canonical-input auditor replaces only that name assertion; all other
checks remain. Original tools and the failed assertion are retained. No worker
was repeated to resolve this audit error.

All four jobs passed. All 72 resource samples pass; peak RSS is 4,721,049,600
bytes. Owner 882890/birth1790206487.24 and all descendants are terminal, code 0.
The selected Core672e5f30/Data065b7a7f and root source81f75c38 remain unchanged.

Closure: bc0942f51e27e70f33ae9fdc994c345fe3c620159478ced1a40d5433777f66d8.
Local evidence: artifacts/parakeet-scalar-where-layout-amd-20260923.
`layout-observations-20260923.json` retains full layouts, fixture pins, results,
and the exact audit reconciliation. Frozen tools: scalar-where-layout-amd at
0e38fdaa; separate reconciliation: audit_canonical_inputs.py at 4c9312a5.

The next isolated candidate will handle uniform masks only, with a float-only
guard, scalar true input and dense false input already matching the broadcast
output. It must validate even unselected operands and allocate independent output.
Mixed masks and unsupported types/layouts retain the complete original path.
This narrower choice is made from captured inputs before candidate timing.
'''
    report = OUT / 'layout-20260923.md'; assert not report.exists()
    report.write_text(text, encoding='utf8', newline='\n')
    print(json.dumps(dict(observations=pin(path), report=pin(report))))


if __name__ == '__main__': main()
