"""Bind the unchanged census consumer and check both actual instruction modes."""
from pathlib import Path
import os
import sys
import common

BASE = Path(__file__).resolve().parent
common.BASE = BASE
pin, read, save, verify, idle, live = common.pin, common.read, common.save, common.verify, common.idle, common.live
DOTNET = common.DOTNET


def build(state,env,spec):
    # Binding phase only: reuse the reviewed consumer and all product binaries.
    (BASE/'runtime').mkdir()
    expected={**spec['runtime_files'],**spec['consumer_files']}
    assert set(expected)==set(spec['runtime_sources'])
    for name,wanted in expected.items():
        path=Path(spec['runtime_sources'][name]);assert pin(path)==wanted,name
        os.link(path,BASE/'runtime'/name)
    save(BASE/'built.json',dict(passed=True,product_rebuilt=False,consumer_rebuilt=False,binding_only=True,
        runtime={p.name:pin(p) for p in (BASE/'runtime').iterdir()}))


def capture(state, env, spec):
    review=read(BASE/'build-review.json');built=read(BASE/'built.json')
    assert review['passed'] and review['built']==pin(BASE/'built.json')
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted
    (BASE/'probe').mkdir()
    for mode in ['512','256']:
        verify()
        for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted
        selected=dict(env)
        if mode=='256':selected['DOTNET_EnableAVX512']='0'
        output=BASE/'probe'/mode
        common.job(state,'census-'+mode,[DOTNET,BASE/'runtime/OwnedWeightCensus.dll',BASE/'spec.json',output,mode],
            selected,BASE,spec['capture_limits'],spec)
        result=read(output/'result.json')
        assert result['passed'] and result['mode']==mode and not result['forced_gc'] and not result['application_scored']
        for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted



if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
