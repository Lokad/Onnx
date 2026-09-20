"""Freeze committed tools and independently verify all original corpus values."""
import argparse, subprocess
from common import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); base.mkdir(parents=True, exist_ok=False)
    ps = psutil_module(); ps.Process().cpu_affinity([0])
    frozen = PRIOR / 'frozen'; reference = frozen / 'reference'; files = {}
    def bind(path, digest=None):
        observed = pin(path)
        if digest is not None: assert observed['sha256'] == digest, str(path)
        files[rel(path)] = observed
    receipt = read(PRIOR / 'receipt.json'); bind(PRIOR / 'receipt.json')
    for name, digest in receipt['evidence'].items(): bind(PRIOR / name, digest)
    bind(frozen / 'source.zip', receipt['source_archive_sha256'])
    build = read(frozen / 'build.json')
    for name, digest in build['frontend'].items(): bind(frozen / 'bin' / name, digest)
    original = read(frozen / 'default-audit.json'); bind(frozen / 'default-audit.json')
    manifest = read(reference / 'manifest.json'); bind(reference / 'manifest.json', original['manifest_sha256'])
    result = read(frozen / 'default.json'); bind(frozen / 'default.json', original['result_sha256'])
    assert manifest['pins'] == read(ROOT / 'tests/pyannote/frontend/pins.json')
    for name, item in manifest['files'].items():
        path = reference / name; bind(path, item['sha256'])
        array = np.load(path, allow_pickle=False)
        assert str(array.dtype) == item['dtype'] and list(array.shape) == item['shape'] and np.isfinite(array).all()
    assert len(manifest['cases']) == len(result['reports']) == 21
    cases = []; values = failures = 0
    for case, row in zip(manifest['cases'], result['reports']):
        assert case['name'] == row['name']
        path = frozen / 'default.json.arrays' / row['file']; bind(path, row['sha256'])
        actual = np.fromfile(path, dtype='<f4').reshape(row['shape'])
        expected = np.load(reference / case['output'], allow_pickle=False)
        comparison = metric(actual, expected, ORIGINAL_LIMIT)
        assert comparison['failed'] == row['bad'] and comparison['max_scaled'] == row['error']
        values += comparison['values']; failures += comparison['failed']
        cases.append(dict(name=case['name'], input=rel(reference / case['input']), native=rel(reference / case['output']),
                          managed=rel(path), shape=row['shape'], original=comparison))
    assert values == 711680 and failures == 3
    subprocess.run(['git', 'diff', '--exit-code', receipt['source'], '--', rel(PRODUCT)], check=True, cwd=ROOT)
    bind(PRODUCT)
    bind(ROOT / 'tests/pyannote/frontend/pins.json')
    tools = Path(__file__).parent
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', rel(tools)], check=True, cwd=ROOT)
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', rel(tools)], cwd=ROOT).strip()
    for path in tools.iterdir():
        if path.is_file(): bind(path)
    tests = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(tools / 'test_routes.py')], capture_output=True, text=True, timeout=60)
    write(base / 'tests.json', dict(code=tests.returncode, stdout=tests.stdout, stderr=tests.stderr))
    assert tests.returncode == 0
    for name in ['window.npy', 'mel.npy']:
        bind(PRIOR / 'reference' / name, pin(reference / name)['sha256'])
    import torch
    assert np.__version__ == '2.2.4' and torch.__version__ == '2.11.0+cpu'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    np.fft.rfft(np.zeros(512)); torch.fft.rfft(torch.zeros(512, dtype=torch.float64))
    numeric = libraries(ps.Process())
    assert numeric
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    spec = dict(source=revision, original_source=receipt['source'], cases=cases, values=values, original_failures=failures,
                files=files, window=rel(reference / 'window.npy'), mel=rel(reference / 'mel.npy'),
                stages=STAGES, limits=LIMITS, reference_limit=REFERENCE_LIMIT, original_limit=ORIGINAL_LIMIT,
                function='Exact saved FP32 coefficients promoted to double; all arithmetic double; preemphasis is exact promoted .97f.',
                tests=pin(base / 'tests.json'), numeric=numeric, interpreter=pin(sys.executable))
    write(base / 'manifest.json', spec)
    print(json.dumps(dict(files=len(files), cases=len(cases), original_values=values, original_failures=failures, manifest=pin(base / 'manifest.json'))))

if __name__ == '__main__': main()
