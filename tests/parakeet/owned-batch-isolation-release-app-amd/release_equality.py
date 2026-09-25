"""Compare retained released-product and relocation outputs directly, without inference."""
import json
from pathlib import Path
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-owned-batch-release-equality-20260925'
RELEASE = ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
CANDIDATE = ROOT/'artifacts/parakeet-owned-batch-isolation-models-amd-20260925'
SOURCES = {'current': (RELEASE, 'selected'), 'candidate': (CANDIDATE, 'candidate')}
DIGESTS = {RELEASE: '5860fc366c3d976907f35f8d7a1832c10566fa08910af32c168dca7a63979842',
    CANDIDATE: '1997eeb782df89975f4893820bc0fc246dec60d1839b2c63a86ad2bba1aecd9d'}


def derive():
    analyses, identities, closures = {}, {}, {}
    for role, (folder, original) in SOURCES.items():
        proof = read(folder/'closed.json')
        assert proof['passed'] and pin(folder/'closed.json')['sha256'] == DIGESTS[folder]
        for name, wanted in proof['files'].items():
            assert pin(folder/name) == wanted, name
        analyses[role] = read(folder/'analysis.json')
        identities[role] = analyses[role]['identities'][original]
        closures[role] = pin(folder/'closed.json')
    assert identities['current']['Lokad.Onnx.dll']['sha256'] == 'f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert identities['candidate']['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert analyses['current']['consumers'] == analyses['candidate']['consumers']
    manifests = {r: read(f/'collected/manifests'/(o+'-parakeet.json')) for r, (f, o) in SOURCES.items()}
    ignored = {'core_sha256', 'data_sha256', 'product_source'}
    assert {k: v for k, v in manifests['current'].items() if k not in ignored} == {
        k: v for k, v in manifests['candidate'].items() if k not in ignored}
    assert pin(RELEASE/'collected/parakeet-reference/manifest.json') == pin(CANDIDATE/'collected/parakeet-reference/manifest.json')
    native, public = [], []
    for isa in ['512', '256']:
        results = {}
        for role, (folder, original) in SOURCES.items():
            path = folder/'collected'/f'{original}-native-{isa}'/'result.json'
            value = read(path)
            proof = analyses[role]['results'][f'{original}-native-{isa}']['native']
            assert proof['audit_consistent'] and proof['numeric_gate_passed'] and proof['application_passed'] and not proof['failures']
            assert proof['result_sha256'] == pin(path)['sha256']
            assert (proof['arrays'], proof['values']) == (784, 3090494)
            assert value['core_sha256'] == identities[role]['Lokad.Onnx.dll']['sha256']
            assert value['data_sha256'] == identities[role]['Lokad.Onnx.Data.dll']['sha256']
            assert value['settings'] == ({} if isa == '512' else {'DOTNET_EnableAVX512': '0'})
            results[role] = (value, path)
        pairs = []
        for a, b in zip(results['current'][0]['rows'], results['candidate'][0]['rows'], strict=True):
            assert a['name'] == b['name'] and a['actual'] == b['actual']
            for x, y in zip(a['comparisons'], b['comparisons'], strict=True):
                assert all(x[k] == y[k] for k in ['label', 'output', 'shape', 'dtype', 'file'])
                left = Path(str(results['current'][1])+'.tensors')/x['file']
                right = Path(str(results['candidate'][1])+'.tensors')/y['file']
                assert pin(left)['sha256'] == x['sha256'] and pin(right)['sha256'] == y['sha256']
                assert left.read_bytes() == right.read_bytes(), (isa, a['name'], x['label'], x['output'])
                pairs.append(dict(case=a['name'], label=x['label'], output=x['output'],
                    values=left.stat().st_size//(8 if x['dtype'] == 'Int64' else 4), sha256=x['sha256'], bit_identical=True))
        assert len(pairs) == 784 and sum(r['values'] for r in pairs) == 3090494
        native.append(dict(isa=isa, arrays=784, values=3090494, comparisons=pairs))
        complete = {}
        for role, (folder, original) in SOURCES.items():
            path = folder/'collected'/f'{original}-public-{isa}'/'output/result.json'
            value = read(path)
            proof = analyses[role]['results'][f'{original}-public-{isa}']
            assert proof['passed'] and proof['public_requests'] == 20 and proof['result'] == pin(path)
            assert value['core_sha256'] == identities[role]['Lokad.Onnx.dll']['sha256']
            assert value['data_sha256'] == identities[role]['Lokad.Onnx.Data.dll']['sha256']
            assert value['held_outputs_unchanged'] and value['flags'] == ({} if isa == '512' else {'DOTNET_EnableAVX512': '0'})
            assert len(value['records']) == 20 and all(r['ownership'] for r in value['records'])
            complete[role] = [dict(name=r['name'], input_sha256=r['input_sha256'], result=r['result']) for r in value['records']]
        assert complete['current'] == complete['candidate'], isa
        public.append(dict(isa=isa, requests=20, complete_results_exact=True))
    return dict(passed=True, identities=identities, consumers=analyses['current']['consumers'],
        source_closures=closures, native=native, public=public, inference_performed=False, performance_scored=False)


def verify():
    proof = read(BASE/'closed.json')
    assert proof['passed'] and proof['generator'] == pin(Path(__file__))
    assert proof['files']['analysis.json'] == pin(BASE/'analysis.json')
    assert read(BASE/'analysis.json') == derive()


if __name__ == '__main__':
    assert not BASE.exists()
    result = derive()
    BASE.mkdir()
    save(BASE/'analysis.json', result)
    save(BASE/'closed.json', dict(passed=True, analysis=pin(BASE/'analysis.json'), generator=pin(Path(__file__)),
        files={'analysis.json': pin(BASE/'analysis.json')}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), array_pairs=1568, value_pairs=6180988,
        public_pairs=40, inference_performed=False)))
