"""Bind unchanged binaries and wait for the unchanged memory guard before each job."""
from pathlib import Path
import os
import sys
import time
import psutil
import common
BASE=Path(__file__).resolve().parent;common.BASE=BASE
pin,read,save,verify,idle,live=common.pin,common.read,common.save,common.verify,common.idle,common.live
DOTNET=common.DOTNET


def runtime_pins():return {p.relative_to(BASE/'runtime').as_posix():pin(p) for p in (BASE/'runtime').rglob('*') if p.is_file()}


def build(state,env,spec):
    first=Path(spec['initial_remote']);receipt=read(first/'capture-collection.json');old=read(first/'capture-state.json')
    assert receipt['terminal'] and receipt['code']==old['code']==1 and old['complete']
    assert pin(first/'capture-collection.json')==spec['initial']['collection'] and pin(first/'capture-state.json')==spec['initial']['state']
    assert not any(live(i) for i in receipt['identities'])
    for name,wanted in spec['binding_runtime'].items():
        source=Path(spec['original_runtime'])/name;assert pin(source)==wanted
        path=BASE/'runtime'/name;path.parent.mkdir(parents=True,exist_ok=True);os.link(source,path)
    save(BASE/'built.json',dict(passed=True,product_rebuilt=False,consumer_rebuilt=False,binding_only=True,runtime=runtime_pins()))


def capture(state,env,spec):
    review=read(BASE/'build-review.json');built=read(BASE/'built.json')
    assert review['passed'] and review['built']==pin(BASE/'built.json') and runtime_pins()==built['runtime']
    assert spec['jobs']==['candidate-512','selected-256','candidate-256']
    for name in spec['jobs']:
        role,mode=name.split('-');verify();assert runtime_pins()==built['runtime']
        # The full-model protocol already permits this wait. Keep every observation.
        started=time.monotonic();observations=[];limits=spec['capture_limits']
        while True:
            value=dict(seconds=time.monotonic()-started,available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(BASE).free)
            observations.append(value);save(BASE/(name+'-preflight-wait.json'),observations)
            assert value['seconds']<spec['preflight_wait_seconds'] and value['tmpfs']>=limits['tmpfs_before']
            if value['available']>=limits['available_before']:break
            time.sleep(10)
        environment=dict(env)
        if mode=='256':environment['DOTNET_EnableAVX512']='0'
        output=BASE/'probe'/name
        common.job(state,name,[DOTNET,BASE/'runtime'/role/'OwnedWeightCounters.dll',BASE/'spec.json',output,role,mode],environment,BASE,limits,spec)
        result=read(output/'result.json');assert result['passed'] and len(result['records'])==20
        assert not result['application_scored'] and not result['forced_gc'] and runtime_pins()==built['runtime']


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
