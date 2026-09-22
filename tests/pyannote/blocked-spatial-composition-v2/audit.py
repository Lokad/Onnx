"""Close fresh normal sources, unchanged product methods and complete suites."""
from common import *
from prepare import SUITES
from transform import transform


def main():
    assert not (BASE/'closed.json').exists(); priors()
    value = read(BASE/'verified.json'); assert value['passed']; verify(value['files'])
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == ['cli-restore', 'cli-build', 'backend-restore', 'backend-build',
        'tensors-restore', 'tensors-build', 'inventory', 'backend-full', 'tensors-full', 'focused-disabled']
    assert all(r['code'] == 0 for r in state['runs'])
    observed, identities = resources(BASE, state, {s[0] for s in SUITES})
    outcomes = [suite(BASE, n, p, s, 0) for n, _, p, s, _, _ in SUITES]
    assert outcomes == value['suites'] and inventory() == value['instruction_review']
    output, diff = transform((FOCUSED/'focused/ConvBlockedSpatialTests.cs').read_text())
    assert (BASE/'source/tests/Lokad.Onnx.Backend.Tests/ConvBlockedSpatialTests.cs').read_text() == output
    assert (BASE/'test-helper.diff').read_text() == diff
    result = dict(passed=True, suites=[dict(name=r['name'], passed=r['passed'], skipped=r['skipped']) for r in outcomes],
        core=value['core'], data=value['data'], resources=observed, instruction_review=inventory(),
        product_source_identical=True, package_qualified=False, models_qualified=False, performance_qualified=False)
    save(BASE/'analysis.json', result)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=value['files'], identities=identities, analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **result)))


if __name__ == '__main__': main()
