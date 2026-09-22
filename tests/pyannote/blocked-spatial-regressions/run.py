"""Run existing regression assemblies against the unchanged isolated product."""
import collections
import importlib.util
import json
from pathlib import Path
import shutil
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-regressions-20260922'
PRODUCT = ROOT/'artifacts/pyannote-blocked-spatial-composition-20260922'
FOCUSED = ROOT/'artifacts/pyannote-blocked-spatial-composition-review-v3-20260922'
LAYERS = ROOT/'artifacts/pyannote-blocked-spatial-layer-graphs-20260922'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('blocked_regression_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal
SUITES = [('backend-existing', 'Backend', 3313, 93), ('tensors', 'Tensors', 343, 0)]


def suite(name, passed, skipped):
    path = BASE/'test-results'/(name+'.trx'); tree = ET.parse(path)
    rows = tree.findall('.//{*}UnitTestResult')
    counts = collections.Counter(r.attrib['outcome'] for r in rows)
    assert counts == dict(Passed=passed, **({'NotExecuted': skipped} if skipped else {})), counts
    counters = tree.find('.//{*}Counters').attrib
    assert int(counters['total']) == passed+skipped and int(counters['passed']) == int(counters['executed']) == passed
    assert int(counters['failed']) == 0
    assert not any('ConvBlockedSpatialTests' in r.attrib['testName'] for r in rows)
    return dict(name=name, passed=passed, skipped=skipped, counters=counters,
                names=[r.attrib['testName'] for r in rows], trx=pin(path))


def prior():
    for folder, filename, sha, passed in [
        (PRODUCT, 'failure-closed.json', 'b52ca1fe7ed0edf06094708ae581c1224cd659a1903ec6844ac58af76856b901', False),
        (FOCUSED, 'closed.json', '23816c9ce3b718bca1a89c81e249c382158f05ef434f4dd6d909bcfb060a1d8a', True),
        (LAYERS, 'closed.json', '3edddec577c52d31d1ec97b952e3ddbd894b26dbe902835cccde958bf59209b4', True)]:
        assert pin(folder/filename)['sha256'] == sha
        proof = read(folder/filename); assert proof['passed'] == passed
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
        for identity in proof.get('identities', []): terminal(identity)
    proof = read(FOCUSED/'analysis.json'); assert proof['passed']
    return proof


def main():
    assert not BASE.exists(); product = prior()
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'test-results').mkdir()
    files = {p.as_posix(): pin(p) for p in [MONITOR, *TOOLS.glob('*'), FOCUSED/'closed.json', LAYERS/'closed.json'] if p.is_file()}
    for _, kind, _, _ in SUITES:
        before = PRODUCT/f'source/tests/Lokad.Onnx.{kind}.Tests/bin/Release/net10.0'
        after = BASE/kind; shutil.copytree(before, after)
        assert pin(after/'Lokad.Onnx.dll') == product['core']
        if (after/'Lokad.Onnx.Data.dll').exists(): assert pin(after/'Lokad.Onnx.Data.dll') == product['data']
        for p in after.rglob('*'):
            if p.is_file(): files[p.as_posix()] = pin(p)
    save(BASE/'inputs.json', dict(files=files, suites=SUITES, core=product['core'], data=product['data'],
        focused_tests_separate=pin(FOCUSED/'closed.json'), test_only_corrections_already_qualified=True))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; outcomes = []
    try:
        for name, kind, passed, skipped in SUITES:
            command = ['dotnet', 'test', BASE/kind/f'Lokad.Onnx.{kind}.Tests.dll', '--tl:off', '--nologo', '-v', 'minimal',
                '--no-build', '--no-restore', '--logger', 'trx;LogFileName='+name+'.trx', '--results-directory', BASE/'test-results']
            if kind == 'Backend': command += ['--filter', 'FullyQualifiedName!~ConvBlockedSpatialTests']
            monitor.worker(state, path, name, command, PRODUCT/'source', [0], 12, 8, 900, True, BASE/'test-results')
            outcomes.append(suite(name, passed, skipped)); save(BASE/'suites.json', outcomes)
            print(json.dumps(dict(name=name, passed=passed, skipped=skipped)), flush=True)
        verify(files); prior()
        save(BASE/'verified.json', dict(passed=True, files=files, suites=outcomes, core=product['core'], data=product['data'],
            existing_backend_tests=3313, corrected_focused_tests_separately=31, existing_backend_skips=93, tensors=343,
            single_combined_suite_not_yet_run=True, no_performance_measurement=True))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
