"""Qualify only the refused instruction mode, reusing the exact reviewed binaries."""
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET
import run as parent

ROOT,TOOLS=parent.ROOT,parent.TOOLS
PRIOR=parent.BASE;REMOTE_PRIOR=parent.REMOTE
BASE=ROOT/'artifacts/parakeet-slice-materialization-mode-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-slice-materialization-mode-20260924'
pin,read,write,ssh=parent.pin,parent.read,parent.write,parent.ssh
transport=parent.transport
PRELUDE=parent.PRELUDE.replace(REMOTE_PRIOR,REMOTE)
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE

REMOTE_SOURCE='''from pathlib import Path
import common
from common import BASE,DOTNET,pin,read,save,live,idle,verify,job

def capture(state,env,spec):
    previous=Path(spec['previous']);built=read(previous/'built.json')
    assert built['core']==spec['core'] and built['consumer']==spec['consumer']
    environment=dict(env,SLICE_CORE_SHA=spec['core']['sha256'],SLICE_AVX512='0',DOTNET_EnableAVX512='0',
        DOTNET_CLI_HOME=str(previous/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',
        NUGET_PACKAGES=str(previous/'packages'),NUGET_HTTP_CACHE_PATH=str(previous/'http-cache'),
        MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0')
    assert 'DOTNET_EnableAVX512F' not in environment
    command=[DOTNET,'test',previous/'source/tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj',
        '-c','Release','--tl:off','--nologo','-v','minimal','--no-build','--no-restore',
        '-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false',
        '-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false',
        '-p:OutputPath='+str(previous/'source/runtime-observed')+'/', '-p:AppendTargetFrameworkToOutputPath=false',
        '--logger','trx;LogFileName=tensors-256.trx','--results-directory',BASE/'logs']
    job(state,'tensors-256',command,environment,previous/'source',spec['capture_limits'],spec)

if __name__=='__main__':
    common.capture=capture
    raise SystemExit(common.main())
'''


def prepare():
    failure=read(PRIOR/'closed.json');assert failure['terminal'] and not failure['passed']
    assert failure['suites'][0]['passed']==369 and not failure['suites'][0]['failed']
    assert failure['suites'][1]['failed']==['Lokad.Onnx.Tensors.Tests.SliceCandidateIdentityTests.ConsumedCoreAndInstructionModeMatch']
    review=read(PRIOR/'build-review.json');assert review['passed'] and failure['build_review']==pin(PRIOR/'build-review.json')
    built=read(PRIOR/'build-collected/built.json');assert review['built']==pin(PRIOR/'build-collected/built.json')
    old=read(PRIOR/'bundle/spec.json');assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    (bundle/'common.py').write_bytes((PRIOR/'bundle/common.py').read_bytes())
    (bundle/'remote.py').write_text(REMOTE_SOURCE,encoding='utf8')
    compile(REMOTE_SOURCE,'corrected-mode-remote','exec')
    external={REMOTE_PRIOR+'/'+name:identity for name,identity in old['files'].items()}
    external.update({REMOTE_PRIOR+'/'+name:identity for name,identity in built['runtime_files'].items()})
    external[REMOTE_PRIOR+'/built.json']=pin(PRIOR/'build-collected/built.json')
    spec=dict(boot=old['boot'],previous=REMOTE_PRIOR,core=built['core'],consumer=built['consumer'],
        external=external,build_limits=old['build_limits'],capture_limits=old['capture_limits'],
        minimum_free=old['minimum_free'],output_limit=old['output_limit'],previous_failure=pin(PRIOR/'closed.json'),
        previous_build_review=pin(PRIOR/'build-review.json'),
        switch_source='https://github.com/dotnet/runtime/blob/v10.0.8/src/coreclr/jit/jitconfigvalues.h#L357',
        files={p.name:pin(p) for p in bundle.iterdir()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.iterdir():archive.add(p,arcname=p.name,recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),controller=pin(__file__)))
    print(json.dumps(dict(prepared=True,core=built['core'],archive=pin(BASE/'payload.tar.gz'))))


def observe():
    assert not (BASE/'closed.json').exists()
    value=ssh(PRELUDE+f'''
from remote import read,live
state=read(base/'capture-state.json') if (base/'capture-state.json').exists() else None
ids=[{read(BASE/'capture-deployment.json')!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},
 error=state and state.get('error'))))
''')
    with (BASE/'observations.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
    print(json.dumps(value))


def audit():
    folder=BASE/'capture-collected';receipt=read(folder/'capture-collection.json')
    assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'capture-state.json');assert state['complete'] and state['code']==0
    run,=state['runs'];assert run['code']==0 and run['complete']
    spec=read(BASE/'bundle/spec.json');limits=spec['capture_limits']
    samples=[json.loads(line) for line in (folder/'logs/tensors-256.resources.jsonl').read_text().splitlines()]
    assert len(samples)==run['samples'] and samples and run['seconds']<limits['seconds']
    assert run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
    for row in samples:
        assert row['seconds']<limits['seconds'] and row['rss']<limits['rss'] and row['output']<spec['output_limit']
        assert row['available']>=spec['minimum_free'] and row['tmpfs']>=spec['minimum_free']
        assert row['rss']==sum(m['rss'] for m in row['members'])
        for member in row['members']:
            assert run['members'][str(member['pid'])]==member['birth'] and member['affinity']==[2] and all(t==[2] for t in member['threads'])
    assert all(0<=b['seconds']-a['seconds']<10 for a,b in zip(samples,samples[1:]))
    ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    suites=[];names=None
    for mode,path in [('512',PRIOR/'capture-collected/logs/tensors-512.trx'),('256',folder/'logs/tensors-256.trx')]:
        if mode=='512':assert pin(path)==read(PRIOR/'closed.json')['suites'][0]['trx']
        tree=ET.parse(path);summary=tree.find('.//t:ResultSummary',ns);assert summary.attrib['outcome']=='Completed'
        counters=summary.find('t:Counters',ns)
        assert int(counters.attrib['total'])==int(counters.attrib['executed'])==int(counters.attrib['passed'])==369
        assert int(counters.attrib['failed'])==int(counters.attrib['notExecuted'])==0
        rows=tree.findall('.//t:UnitTestResult',ns);assert len(rows)==369 and all(r.attrib['outcome']=='Passed' for r in rows)
        current=sorted(r.attrib['testName'] for r in rows);assert len(set(current))==369
        if names is None:names=current
        else:assert current==names
        assert sum('SliceReshapeCopyTests.' in n for n in current)==25
        assert sum('SliceCandidateIdentityTests.' in n for n in current)==1
        suites.append(dict(mode=mode,passed=369,skipped=0,trx=pin(path)))
    result=dict(passed=True,core=spec['core'],consumer=spec['consumer'],suites=suites,same_binaries=True,
        resources=dict(samples=len(samples),peak_rss=max(r['rss'] for r in samples)),
        previous_failure=spec['previous_failure'],build_review=spec['previous_build_review'],performance_measured=False)
    write(BASE/'analysis.json',result)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),collection=pin(folder/'capture-collection.json'),
        previous_failure=spec['previous_failure'],auditor=pin(__file__),terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for p,b in run['members'].items()]))
    print(json.dumps(result))


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='stage':transport.stage()
    elif action=='launch':transport.launch('capture')
    elif action=='observe':observe()
    elif action=='collect':transport.collect('capture')
    elif action=='audit':audit()
    else:raise ValueError(action)
