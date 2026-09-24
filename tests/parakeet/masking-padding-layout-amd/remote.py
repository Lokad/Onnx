"""Reuse the bounded Data build; observe the original 80 complete requests."""
import importlib.util
from pathlib import Path
import common
from common import BASE, DOTNET, pin, read, save, live, idle, verify, job


def capture(state, env, spec):
    review = read(BASE/'build-review.json'); assert review['passed'] and review['built'] == pin(BASE/'built.json')
    built = read(BASE/'built.json')
    assert built['core'] == spec['core']
    for name, wanted in built['runtime_files'].items(): assert pin(BASE/name) == wanted, name
    app = Path(spec['app']); runtime = BASE/'runtime-observed'; output = BASE/'phase'
    module = importlib.util.spec_from_file_location('cpu_accounting', app/'runtime/campaign_processes.py')
    accounting = importlib.util.module_from_spec(module); module.loader.exec_module(accounting)
    env = {k:v for k,v in env.items() if not k.lower().startswith('parakeet_masking_')}
    env['PARAKEET_MASKING_DATA_SHA'] = built['data']['sha256']
    before = accounting.snapshot()
    job(state,'phase',[DOTNET,runtime/'SampledAudio.dll',app/'assets',app/spec['manifest'],output,'timing','control'],
        env,BASE,spec['capture_limits'],spec,output)
    after = accounting.snapshot(); row = state['runs'][-1]
    row['cpu_before'] = before; row['cpu_after'] = after
    row['accounting'] = accounting.foreign_fraction(before,after,state['supervisor']['pid'])
    assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction'] <= .01
    save(BASE/'capture-state.json',state)
    result = read(output/'result.json'); assert result['passed'] and len(result['records']) == 80
    assert len(list(output.glob('layout-*.json'))) == 80


if __name__ == '__main__':
    common.capture = capture
    raise SystemExit(common.main())
