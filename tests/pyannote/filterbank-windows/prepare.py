"""Bind every original trace and extract the complete fixed 32-window scope."""
import argparse, math, subprocess
from shared import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); base.mkdir(parents=True, exist_ok=False); (base / 'inputs').mkdir()
    psutil_module().Process().cpu_affinity([0]); files = {}
    def bind(path, digest=None):
        observed = pin(path)
        if digest is not None: assert observed['sha256'] == digest, str(path)
        files[rel(path)] = observed
        return observed
    bind(PRIOR_REFERENCE / 'closed.json', REFERENCE_RECEIPT)
    prior_receipt = read(PRIOR_REFERENCE / 'closed.json'); prior = read(PRIOR_REFERENCE / 'manifest.json')
    for name, wanted in prior_receipt['files'].items(): assert bind(PRIOR_REFERENCE / name) == wanted
    for name, wanted in prior_receipt['reports'].items(): assert bind(ROOT / name) == wanted
    for name, wanted in prior['files'].items(): assert bind(ROOT / name) == wanted
    for name, wanted in prior['numeric'].items(): assert pin(name) == wanted
    assert prior['interpreter'] == pin(sys.executable)
    cases = []; trace_summaries = []; omissions = []; original_revisions = {}
    for corpus in CORPORA:
        folder = ROOT / corpus['directory']; frozen = folder / corpus['frozen']; reference = frozen / 'reference'
        receipt = read(folder / 'receipt.json'); bind(folder / 'receipt.json')
        for group in ['files', 'evidence', 'recipes']:
            for name, digest in receipt.get(group, {}).items(): bind(folder / name, digest)
        if corpus['name'] == 'dialogue':
            all_hosts = read(folder / 'receipt-all-hosts.json'); bind(folder / 'receipt-all-hosts.json')
            for name, digest in all_hosts['files'].items(): bind(folder / name, digest)
        build = read(frozen / 'build.json'); bind(frozen / 'build.json')
        archive = frozen / 'source.zip' if corpus['name'] == 'five' else folder / 'archive/source.zip'
        bind(archive, receipt['source_archive_sha256'])
        result_digest = all_hosts['amd_result_archive_sha256'] if corpus['name'] == 'dialogue' else receipt['result_archive_sha256']
        bind(folder / 'amd-results.tar.gz', result_digest)
        binaries = build.get('files', build.get('binaries'))
        for name, wanted in binaries.items(): assert bind(frozen / 'bin' / name) == wanted
        original_revisions[corpus['name']] = build['source']
        subprocess.run(['git', 'diff', '--exit-code', build['source'], '--', 'src/Lokad.Onnx.Data/WeSpeakerAudio.cs', 'src/Lokad.Onnx.Data/Community1Timeline.cs'], cwd=ROOT, check=True)
        manifest = read(reference / 'manifest.json'); manifest_pin = bind(reference / 'manifest.json')
        assert [len(c['windows']) for c in manifest['cases']] == corpus['case_windows']
        for name, wanted in manifest['files'].items():
            path = reference / name; bind(path, wanted['sha256']); value = np.load(path, allow_pickle=False)
            assert list(value.shape) == wanted['shape'] and str(value.dtype) == wanted['dtype'] and np.isfinite(value).all()
        traces = {}; indexes = {}
        for host, location in [('windows', frozen), ('amd', folder / 'amd-collected/result')]:
            for config in ['default', 'preferred']:
                variant = host + '-' + config; detail = location / (config + '-detail')
                trace = read(detail / 'result.json'); trace_pin = bind(detail / 'result.json')
                auditpath = frozen / (config + '-audit.json') if host == 'windows' else folder / ('amd-' + config + '-audit.json')
                audit = read(auditpath); bind(auditpath)
                assert audit['detail_sha256'] == trace_pin['sha256'] and audit['reference_sha256'] == manifest_pin['sha256'] == trace['reference_sha256']
                for name, observed in trace['assemblies'].items():
                    assert binaries[name + '.dll']['sha256'] == observed['sha256'] and build['source'] in observed['version']
                assert trace['execution_complete'] and len(trace['reports']) == len(audit['numerical']['rows'])
                failures = total_failures = values = 0; maximum = 0.
                for row, old in zip(trace['reports'], audit['numerical']['rows']):
                    path = detail / row['file']; bind(path, row['sha256'])
                    target = np.load(reference / row['reference'], allow_pickle=False)
                    actual = np.fromfile(path, dtype='<f4').reshape(target.shape)
                    check = metric(actual, target, ORIGINAL_LIMIT)
                    assert (row['name'], row['stage']) == (old['name'], old['stage']) and row['length'] == check['values'] == old['values']
                    assert check['failed'] == old['bad'] and math.isclose(check['max_scaled'], old['maximum'], rel_tol=1e-14)
                    total_failures += check['failed']; maximum = max(maximum, check['max_scaled'])
                    if row['stage'] == 'features': failures += check['failed']; values += check['values']
                assert total_failures == audit['numerical']['bad'] and maximum == audit['numerical']['maximum']
                assert failures == corpus['failures'][host] and values == corpus['windows'] * 79840
                trace_summaries.append(dict(corpus=corpus['name'], variant=variant, arrays=len(trace['reports']), total_failures=total_failures, feature_failures=failures, feature_values=values))
                traces[variant] = trace; indexes[variant] = (detail, feature_inventory(manifest, trace))
            assert traces[host + '-default']['reports'] == traces[host + '-preferred']['reports']
        for case in manifest['cases']:
            pcm_path = reference / case['pcm']; pcm = np.load(pcm_path, allow_pickle=False)
            for index, window in enumerate(case['windows']):
                if 'features' not in window:
                    omissions.append(dict(corpus=corpus['name'], name=case['name'], window=index, reason='Original trace has no filterbank'))
                    continue
                name = corpus['name'] + '-' + case['name'] + f'-w{index:02}'
                data, valid = extract(pcm, index); path = base / 'inputs' / (name + '.npy')
                with path.open('xb') as stream: np.save(stream, data, allow_pickle=False)
                bind(path); native_path = reference / window['features']; native = np.load(native_path, allow_pickle=False)
                assert native.shape == (1, 998, 80)
                baselines = dict(native=dict(file=rel(native_path), pin=pin(native_path), shape=list(native.shape), format='npy'))
                original = {}
                for variant in VARIANTS[1:]:
                    detail, lookup = indexes[variant]; row = lookup[(case['name'], window['features'])]; location = detail / row['file']
                    baselines[variant] = dict(file=rel(location), pin=pin(location), shape=list(native.shape), format='raw')
                    original[variant] = metric(load_baseline(baselines[variant]), native, ORIGINAL_LIMIT)
                cases.append(dict(name=name, corpus=corpus['name'], original_case=case['name'], window=index, valid_samples=valid,
                                  original_pcm=rel(pcm_path), input=rel(path), baselines=baselines, original=original))
    assert len(cases) == 32 and omissions == [dict(corpus='five', name='silence', window=0, reason='Original trace has no filterbank')]
    for path in [PRODUCT, ROOT / 'src/Lokad.Onnx.Data/Community1Timeline.cs']: bind(path)
    folder = Path(__file__).parent
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', rel(folder)], cwd=ROOT, check=True)
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', rel(folder)], cwd=ROOT).strip()
    for path in folder.iterdir():
        if path.is_file(): bind(path)
    tests = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(folder / 'test_windows.py')], capture_output=True, text=True, timeout=60)
    write(base / 'tests.json', dict(code=tests.returncode, stdout=tests.stdout, stderr=tests.stderr)); assert tests.returncode == 0
    spec = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), original_revisions=original_revisions,
                cases=cases, omissions=omissions, variants=VARIANTS, values=2554880, files=files, trace_summaries=trace_summaries,
                window=prior['window'], mel=prior['mel'], stages=STAGES, limits=LIMITS, reference_limit=REFERENCE_LIMIT,
                original_limit=ORIGINAL_LIMIT, function=prior['function'], numeric=prior['numeric'], interpreter=prior['interpreter'], tests=pin(base / 'tests.json'))
    write(base / 'manifest.json', spec)
    print(json.dumps(dict(files=len(files), cases=len(cases), values=spec['values'], original=trace_summaries, manifest=pin(base / 'manifest.json'))))

if __name__ == '__main__': main()
