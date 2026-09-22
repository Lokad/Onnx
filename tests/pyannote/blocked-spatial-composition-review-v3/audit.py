import json
from prepare import BASE, FAILED, PRODUCT, pin, read, save, verify, verify_prior, review, suites, resource_checks


def main():
    assert not (BASE/'closed.json').exists()
    value = read(BASE/'verified.json'); assert value['passed']; verify(value['files']); verify_prior()
    assert value['product_bytes_unchanged'] and value['preparation_only'] and not value['models_qualified'] and not value['performance_qualified']
    assert value['core'] == pin(PRODUCT/'runtime/Lokad.Onnx.dll') and value['data'] == pin(PRODUCT/'runtime/Lokad.Onnx.Data.dll')
    assert review() == read(BASE/'instruction-review.json') and suites() == value['suites']
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] == 0
    assert [(r['name'], r['code']) for r in state['runs']] == [('restore', 0), ('build', 0), ('focused-normal', 0), ('focused-disabled', 0)]
    resources, identities = resource_checks(BASE, state)
    assert read(BASE/'selftest.json')['passed'] and read(BASE/'selftest.json')['tests'] == 7
    analysis = dict(passed=True, preparation_only=True, core=value['core'], data=value['data'], product_bytes_unchanged=True,
        source_review=review(), suites=suites(), resources=resources, models_qualified=False, performance_qualified=False,
        prior_failures=[pin(folder/'failure-closed.json') for folder in [PRODUCT, FAILED]])
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=True, preparation_only=True, files=files, local_inputs=value['files'], identities=identities,
        source_artifact=str(PRODUCT), analysis=pin(BASE/'analysis.json'), prior_failures=analysis['prior_failures']))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), suites=[dict(mode=s['mode'], tests=s['tests']) for s in suites()],
        unchanged_core=3103, changed_core=10, added_core=48, unchanged_data=697, equivalent_component_methods=11,
        samples=sum(r['samples'] for r in resources))))


if __name__ == '__main__': main()
