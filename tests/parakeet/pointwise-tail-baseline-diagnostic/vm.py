"""Observe exact baseline versus itself; no compilation or performance trial."""
from pathlib import Path
import sys
import common

BASE=Path(__file__).resolve().parent;common.BASE=BASE
pin,read,save,verify,idle,live=common.pin,common.read,common.save,common.verify,common.idle,common.live


def capture(state,env,spec):
    for mode in spec['modes']:
        environment=dict(env,DOTNET_JitDisasm=spec['disasm'])
        if mode=='avx512-disabled':environment['DOTNET_EnableAVX512']='0'
        common.job(state,mode,[common.DOTNET,BASE/'runtime/TailContracts.dll',BASE/'spec.json',BASE/'built.json',
            BASE/'probe'/mode,mode],environment,BASE,spec['capture_limits'],spec)
        assert read(BASE/'probe'/mode/'result.json')['completed']


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize and sys.argv[1]=='capture'
    common.capture=capture
    raise SystemExit(common.main())
