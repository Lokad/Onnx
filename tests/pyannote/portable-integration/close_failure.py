"""Close the metadata-counter refusal and freeze mutable root input snapshots."""
import shutil
from phase_v2 import *


def main():
    assert not (BASE / 'failure-closed.json').exists()
    state = read(BASE / 'processes.json')
    assert state['complete'] and state['code'] == 1 and state['error'].endswith('AssertionError\n')
    assert "counters['notExecuted']" in state['error']
    names = ['cli-restore', 'cli-build', 'backend-restore', 'backend-build', 'tensors-restore', 'tensors-build',
        'bridge-restore', 'bridge-build', 'instructions', 'focused', 'hardware-disabled', 'backend-full']
    evidence = resources(BASE, names, 1)
    suites = [read_suite(BASE, *args) for args in [('focused', 203, 0), ('hardware-disabled', 109, 0), ('backend-full', 3290, 93)]]
    assert suites[-1]['counters']['notExecuted'] == '0' and suites[-1]['outcomes']['NotExecuted'] == 93
    prepared = read(BASE / 'source-prepared.json')
    verify(prepared['files'])
    assert read(BASE / 'instructions.json')['passed']
    frozen = {}
    for name, wanted in prepared['files'].items():
        if name.startswith('artifacts/'):
            continue
        target = BASE / 'evidence-inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        assert pin(target) == wanted
        frozen[name] = dict(path=rel(target), **wanted)
    save(BASE / 'root-input-snapshots.json', frozen)
    # This closure uses byte-identical snapshots, so later authorized product
    # integration need not change historical input evidence or old manifests.
    files = {name: wanted for name, wanted in prepared['files'].items() if name.startswith('artifacts/')}
    for path in [*BASE.rglob('*'), TOOLS / 'close_failure.py', TOOLS / 'phase_v2.py']:
        if path.is_file() and not {'obj', 'nuget-cache', 'consumer-cache'}.intersection(path.relative_to(ROOT).parts):
            files[rel(path)] = pin(path)
    save(BASE / 'failure-analysis.json', dict(passed=False, numerical_failures=0, checker_refusal=True,
        reason='Individual TRX records contain 93 skipped cases, but its aggregate notExecuted is zero. Every executed backend test passed.',
        suites=suites, frozen_root_inputs=frozen, **evidence,
        scope='Preserve the failed checker; reuse completed builds, method/API proof and successful suites. Tensors and package work have not run.'))
    files[rel(BASE / 'failure-analysis.json')] = pin(BASE / 'failure-analysis.json')
    verify(files)
    save(BASE / 'failure-closed.json', dict(passed=False, checker_refusal=True, numerical_failures=0,
        files=files, analysis=pin(BASE / 'failure-analysis.json'), identities=evidence['identities']))
    print(json.dumps(dict(failure=pin(BASE / 'failure-closed.json'), resources=evidence['resource_samples'], identities=len(evidence['identities']))))


if __name__ == '__main__':
    main()
