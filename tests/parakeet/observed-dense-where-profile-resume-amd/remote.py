"""Reuse the exact reviewed observer and bounded matched-profile capture."""
import importlib.util
from pathlib import Path
import sys
import common
from common import BASE, DOTNET, idle, job, live, pin, read, save, verify


def capture(state,env,spec):
    approval=read(BASE/'observer-review.json');assert approval['passed'] and approval['runtime']==pin(BASE/'runtime.json')
    built=read(BASE/'runtime.json')
    for name,wanted in built['runtime_files'].items():assert pin(BASE/name)==wanted
    app=Path(spec['app'])
    module=importlib.util.spec_from_file_location('cpu_accounting',app/'runtime/campaign_processes.py')
    accounting=importlib.util.module_from_spec(module);module.loader.exec_module(accounting)
    for name in ['wall']:
        verify();runtime=BASE/('runtime-control' if name=='control' else 'runtime-observed');output=BASE/name
        environment=dict(env,PARAKEET_PHASE_MODE='wall',PARAKEET_PHASE_DATA_SHA=spec['data']['sha256'],
            PARAKEET_PHASE_CORE_SHA=spec['core' if name=='control' else 'candidate_core']['sha256'])
        before=accounting.snapshot()
        job(state,name,[DOTNET,runtime/'SampledAudio.dll',app/'assets',app/spec['manifest'],
            output,'timing','control'],environment,BASE,spec['capture_limits'],spec,output)
        after=accounting.snapshot();row=state['runs'][-1]
        row.update(cpu_before=before,cpu_after=after,
            accounting=accounting.foreign_fraction(before,after,state['supervisor']['pid']))
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction']<=.01
        save(BASE/'capture-state.json',state)
        value=read(output/'result.json');assert value['passed'] and len(value['records'])==80


if __name__=='__main__':
    assert sys.argv[1:]==['capture']
    common.capture=capture
    raise SystemExit(common.main())
