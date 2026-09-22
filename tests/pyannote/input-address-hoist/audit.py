"""Close terminal local proof, without claiming the unexecuted AVX512 path."""
import importlib.util
import json
from build import ROOT, BASE, SOURCE, pin, read, save, verify, product_review, consumer_review, results
from transform import transform


def main():
    assert not (BASE/'closed.json').exists()
    proof = read(BASE/'verified.json'); assert proof['passed'] and not proof['avx512_qualified'] and proof['no_performance_measurement']
    for name in ['inputs.json', 'consumer-inputs.json']: verify(read(BASE/name)['files'])
    assert product_review() == proof['product'] and results() == proof['reports']
    assert [consumer_review(mode) for mode in ['raw', 'wide', 'layers']] == proof['consumers']
    import tarfile
    with tarfile.open(BASE/'source.tar') as tar:
        stream = tar.extractfile(SOURCE); assert stream is not None
        original = stream.read().decode()
    candidate, diff = transform(original)
    assert (BASE/'source'/SOURCE).read_text() == candidate
    assert (BASE/'candidate.patch').read_text() == diff
    common_path = ROOT/'tests/parakeet/portable-models/common.py'
    spec = importlib.util.spec_from_file_location('input_address_resource_audit', common_path)
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    jobs = {'cli-restore': [8, 8, 900, False], 'cli-build': [8, 8, 900, False], 'inventory': [8, 8, 900, False]}
    for mode in ['raw', 'wide', 'layers']:
        for action in ['restore', 'build', 'inventory']: jobs[mode+'-'+action] = [8, 8, 900, False]
    for mode in ['raw', 'wide', 'layers']: jobs[mode+'-256'] = [12, 8, 900, True]
    assert read(BASE/'jobs.json') == jobs
    resources = common.resources(BASE, 'controller.json', jobs)
    state = read(BASE/'controller.json')
    for mode in ['raw', 'wide', 'layers']:
        report = read(BASE/'output'/(mode+'-256.json'))
        row, = [r for r in state['runs'] if r['name'] == mode+'-256']
        assembly = 'LayerGraphs.dll' if mode == 'layers' else 'Lokad.Onnx.Backend.Tests.dll'
        assert report['pid'] == row['worker']['pid'] and report['executable'] == pin(BASE/'consumers'/mode/'bin/Release/net10.0'/assembly)['sha256']
    analysis = dict(**proof, resources=resources['resources'])
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=True, files=files, identities=resources['identities'], no_performance_measurement=True, avx512_qualified=False))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), core=proof['core'], reports=proof['reports'], resources=sum(r['samples'] for r in resources['resources']))))


if __name__ == '__main__': main()
