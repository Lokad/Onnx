"""Build unchanged-Core observation, then control and observed full applications."""
import importlib.util
from pathlib import Path
import common
from common import BASE, DOTNET, pin, read, save, job, live, idle, verify


def capture(state, env, spec):
    review = read(BASE/'build-review.json')
    assert review['passed'] and review['built'] == pin(BASE/'built.json')
    built = read(BASE/'built.json'); assert built['core'] == spec['core']
    for name,wanted in built['runtime_files'].items(): assert pin(BASE/name) == wanted,name
    app = Path(spec['app']); runtime = BASE/'runtime-observed'
    module = importlib.util.spec_from_file_location('cpu_accounting',app/'runtime/campaign_processes.py')
    accounting = importlib.util.module_from_spec(module); module.loader.exec_module(accounting)
    env = {k:v for k,v in env.items() if not k.lower().startswith('parakeet_projection_')}
    env['PARAKEET_PROJECTION_DATA_SHA'] = built['data']['sha256']
    for mode in ['control','phase']:
        common.verify(); output = BASE/mode
        environment = dict(env,PARAKEET_PROJECTION_MODE=mode)
        before = accounting.snapshot()
        job(state,mode,[DOTNET,runtime/'SampledAudio.dll',app/'assets',app/spec['manifest'],output,'timing','control'],
            environment,BASE,spec['capture_limits'],spec,output)
        after = accounting.snapshot(); row = state['runs'][-1]
        row['cpu_before'] = before; row['cpu_after'] = after
        row['accounting'] = accounting.foreign_fraction(before,after,state['supervisor']['pid'])
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction'] <= .01
        save(BASE/'capture-state.json',state)
        result = read(output/'result.json'); assert result['passed'] and len(result['records']) == 80
        assert len(list(output.glob('projection-*.json'))) == 80


if __name__ == '__main__':
    common.capture = capture
    raise SystemExit(common.main())
