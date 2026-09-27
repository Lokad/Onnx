"""Original bounded build, followed by unchanged control and scoped observer."""
from base_build import *


def capture(state, env, spec):
    review = read(BASE/'build-review.json')
    built = read(BASE/'built.json')
    assert review['passed'] and review['built'] == pin(BASE/'built.json')
    for name, wanted in built['runtime'].items(): assert pin(BASE/'runtime'/name) == wanted
    app = Path(spec['app'])
    for role in ['control', 'observed']:
        common.verify()
        runtime = Path(spec['prior']) if role == 'control' else BASE/'runtime'
        output = BASE/'probe'/role
        environment = dict(env, PARAKEET_PHASE_MODE='control',
            PARAKEET_PHASE_CORE_SHA=pin(runtime/'Lokad.Onnx.dll')['sha256'],
            PARAKEET_PHASE_DATA_SHA=pin(runtime/'Lokad.Onnx.Data.dll')['sha256'])
        environment.pop('PARAKEET_ATTENTION_COSTS', None)
        if role == 'observed': environment['PARAKEET_ATTENTION_COSTS'] = str(BASE/'logs/costs.json')
        before = accounting.snapshot()
        common.job(state, role, [DOTNET, runtime/'SampledAudio.dll', app/'assets',
            app/'manifests/current-parakeet.json', output, 'timing', 'control'],
            environment, BASE, spec['capture_limits'], spec, output)
        after = accounting.snapshot()
        row = state['runs'][-1]
        row['cpu_before'], row['cpu_after'] = before, after
        row['accounting'] = accounting.foreign_fraction(before, after, state['supervisor']['pid'])
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction'] <= .01
        save(BASE/'capture-state.json', state)
        assert read(output/'result.json')['passed']
    assert read(BASE/'logs/costs.json')['passed']
    for name, wanted in built['runtime'].items(): assert pin(BASE/'runtime'/name) == wanted


if __name__ == '__main__':
    assert sys.platform == 'linux' and not sys.flags.optimize
    common.build, common.capture = build, capture
    raise SystemExit(common.main())
