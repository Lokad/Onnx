"""Correct the test helper's explicit dictionary type in a fresh source build."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PREVIOUS = ROOT/'tests/pyannote/blocked-spatial-composition-v2'
sys.path.insert(0, str(PREVIOUS))
import common as shared
import prepare as previous
import transform as previous_transform

FAILED = previous.BASE
BASE = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922'
previous.BASE = shared.BASE = shared.monitor.BASE = BASE
previous.TOOLS = shared.TOOLS = TOOLS
pin, read, save, verify, terminal = shared.pin, shared.read, shared.save, shared.verify, shared.terminal
old_priors = shared.priors
old_transform = previous_transform.transform


def transform(corrected):
    output, diff = old_transform(corrected)
    old = 'graph.Execute(new() { ["x"] = graph.Inputs["x"] }, true, ExecutionProvider.CPU, options)'
    new = 'graph.Execute(new Dictionary<string, ITensor> { ["x"] = graph.Inputs["x"] }, true, ExecutionProvider.CPU, options)'
    assert output.count(old) == 1
    output = output.replace(old, new)
    import difflib
    diff = ''.join(difflib.unified_diff(corrected.splitlines(True), output.splitlines(True),
        fromfile='corrected-api-consumer', tofile='explicit-overloads-and-dictionary-type'))
    return output, diff


def close_failure():
    assert not (FAILED/'failure-closed.json').exists()
    value = read(FAILED/'inputs.json'); verify(value['files'])
    state = read(FAILED/'controller.json'); assert state['complete'] and state['code'] == 1
    assert [(r['name'], r['code']) for r in state['runs']] == [('cli-restore', 0), ('cli-build', 0), ('backend-restore', 0), ('backend-build', 1)]
    assert not list((FAILED/'test-results').glob('*.trx'))
    log = (FAILED/'logs/backend-build.log').read_text()
    assert "error CS0021: Cannot apply indexing with [] to an expression of type 'object'" in log
    observed, identities = shared.resources(FAILED, state, set())
    save(FAILED/'failure-analysis.json', dict(passed=False, retained_failure=True, resources=observed,
        cause='New test helper target-typed new selected the object Execute overload; explicit Dictionary required',
        product_projects_built=True, tests_executed=False))
    files = {p.relative_to(FAILED).as_posix(): pin(p) for p in FAILED.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(FAILED).parts)}
    save(FAILED/'failure-closed.json', dict(passed=False, retained_failure=True, files=files, identities=identities, local_inputs=value['files']))


def priors():
    old_priors()
    value = read(FAILED/'failure-closed.json'); assert value['retained_failure'] and not value['passed']
    for name, wanted in value['files'].items(): assert pin(FAILED/name) == wanted, name
    for identity in value['identities']: terminal(identity)


previous.transform = previous_transform.transform = transform
previous.close_failure = close_failure
previous.priors = shared.priors = priors


if __name__ == '__main__': previous.main()
