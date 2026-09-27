"""Reuse consumer-only build, then serial timing with original CPU accounting."""
from base_build import *
import campaign_processes as accounting


def capture(state,env,spec):
    built=read(BASE/'built.json')
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted
    for job in spec['jobs']:
        verify();before=accounting.snapshot()
        common.job(state,job['name'],[DOTNET,BASE/'runtime'/job['role']/'TailContracts.dll',BASE/'spec.json',BASE/'built.json',
            BASE/'probe'/job['name'],job['role']],env,BASE,spec['capture_limits'],spec)
        after=accounting.snapshot();row=state['runs'][-1]
        row['cpu_before'],row['cpu_after']=before,after
        row['accounting']=accounting.foreign_fraction(before,after,state['supervisor']['pid'])
        save(BASE/'capture-state.json',state)
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction']<=spec['foreign_cpu_limit']
        assert read(BASE/'probe'/job['name']/'result.json')['passed']
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
