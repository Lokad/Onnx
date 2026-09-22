"""Retain the test-contract failure and qualify explicit cache refresh semantics."""
import difflib
import json
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
V2TOOLS = ROOT/'tests/pyannote/blocked-spatial-composition-review-v2'
sys.path.insert(0, str(V2TOOLS))
import run as previous

BASE = ROOT/'artifacts/pyannote-blocked-spatial-composition-review-v3-20260922'
FAILED = previous.BASE
PRODUCT = previous.PRIOR
previous.BASE = BASE; previous.monitor.BASE = BASE
monitor = previous.monitor
pin, read, save, verify, terminal = previous.pin, previous.read, previous.save, previous.verify, previous.terminal
review, suites = previous.review, previous.suites


def inspect_failed():
    assert not (FAILED/'failure-closed.json').exists()
    value = read(FAILED/'inputs.json'); verify(value['files'])
    state = read(FAILED/'controller.json'); assert state['complete'] and state['code'] == 1
    assert [(r['name'], r['code']) for r in state['runs']] == [('restore', 0), ('build', 0), ('focused-normal', 1)]
    resources, identities = resource_checks(FAILED, state)
    ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    tree = ET.parse(FAILED/'test-results/focused-normal.trx')
    counters = tree.find('.//t:Counters', ns).attrib
    assert (counters['total'], counters['passed'], counters['failed']) == ('31', '30', '1')
    rows = tree.findall('.//t:UnitTestResult', ns); assert len(rows) == 31
    failed, = [r for r in rows if r.attrib['outcome'] != 'Passed']
    assert failed.attrib['testName'].endswith('.ReplacementAndExplicitMutationInvalidationRefreshValues')
    message = failed.find('.//t:Message', ns).text
    assert message == 'Assert.NotSame() Failure: Values are the same instance'
    assert review() == read(FAILED/'instruction-review.json')
    save(FAILED/'failure-analysis.json', dict(passed=False, retained_failure=True, tests=31, passed_tests=30,
        failed_test=failed.attrib['testName'], message=message, product_numerical_mismatch_observed=False,
        source_review_passed=True, resources=resources, error=state['error']))
    files = {p.relative_to(FAILED).as_posix(): pin(p) for p in FAILED.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(FAILED).parts)}
    save(FAILED/'failure-closed.json', dict(passed=False, retained_failure=True, files=files, local_inputs=value['files'], identities=identities,
        core=pin(PRODUCT/'runtime/Lokad.Onnx.dll'), data=pin(PRODUCT/'runtime/Lokad.Onnx.Data.dll')))
    return read(FAILED/'failure-closed.json')


def resource_checks(folder, state):
    identities = [state['supervisor']]; resources = []
    for row in state['runs']:
        assert row['complete'] and row['seconds'] < 900
        assert row['preflight']['available'] >= (12 if row['name'].startswith('focused-') else 8)*1024**3
        samples = [json.loads(s) for s in (folder/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for s in samples:
            assert s['seconds'] < 900 and s['rss'] < 8*1024**3 and s['available'] >= 1024**3
            assert s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3 and s['rss'] == sum(m['rss'] for m in s['members'])
            for m in s['members']: assert m['affinity'] == [2] and row['members'][str(m['pid'])] == m['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: terminal(identity)
    return resources, identities


def verify_prior():
    for folder in [PRODUCT, FAILED]:
        value = read(folder/'failure-closed.json'); assert value['retained_failure'] and not value['passed']
        for name, wanted in value['files'].items(): assert pin(folder/name) == wanted, name
        for identity in value['identities']: terminal(identity)


def main():
    assert not BASE.exists()
    prior = inspect_failed(); verify_prior()
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'test-results').mkdir(); source = BASE/'focused'; source.mkdir()
    save(BASE/'instruction-review.json', review())
    shutil.copy2(FAILED/'selftest.json', BASE/'selftest.json')
    before = (FAILED/'focused/ConvBlockedSpatialTests.cs').read_text()
    old = '        if (old is not null) Assert.NotSame(old, Assert.Single(graph.PackedConvWeights).Value);'
    new = '''        // Replacement is safe through fallback; explicit preparation rebuilds the clone.
        graph.Prepare();
        if (old is not null) Assert.NotSame(old, Assert.Single(graph.PackedConvWeights).Value);
        Equal(Reference(graph), Run(graph).ToArray());'''
    assert before.count(old) == 1; after = before.replace(old, new)
    (source/'ConvBlockedSpatialTests.cs').write_text(after, encoding='utf8')
    (BASE/'explicit-refresh.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='failed-cache-assertion', tofile='documented-explicit-refresh')), encoding='utf8')
    project = source/'Focused.csproj'; shutil.copy2(FAILED/'focused/Focused.csproj', project)
    files = {p.as_posix(): pin(p) for p in [*TOOLS.glob('*'), *V2TOOLS.glob('*'), *source.glob('*'), previous.MONITOR,
        PRODUCT/'failure-closed.json', FAILED/'failure-closed.json', BASE/'explicit-refresh.diff', BASE/'instruction-review.json'] if p.is_file()}
    save(BASE/'inputs.json', dict(files=files, core=prior['core'], data=prior['data'], product_bytes_unchanged=True))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; save(path, state); flags = monitor.FLAGS+['-p:NuGetAudit=false']
    try:
        def run(name, command, numerical=False):
            monitor.worker(state, path, name, command, ROOT, [0], 12 if numerical else 8, 8, 900, True, source)
            print(name, 'passed', flush=True)
        run('restore', ['dotnet', 'restore', project, *flags, '--source', previous.FEED, '--packages', BASE/'packages'])
        run('build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        assert pin(source/'bin/Release/net10.0/Lokad.Onnx.dll') == prior['core']
        for mode in ['normal', 'disabled']:
            clean = monitor.clean_env
            if mode == 'disabled': monitor.clean_env = lambda: clean() | {'DOTNET_EnableHWIntrinsic': '0'}
            try:
                run('focused-'+mode, ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                    '--logger', 'trx;LogFileName=focused-'+mode+'.trx', '--results-directory', BASE/'test-results'], True)
            finally: monitor.clean_env = clean
        verify_prior(); verify(files)
        save(BASE/'verified.json', dict(passed=True, files=files, suites=suites(), core=prior['core'], data=prior['data'],
            product_bytes_unchanged=True, preparation_only=True, models_qualified=False, performance_qualified=False))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
