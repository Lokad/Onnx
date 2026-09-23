"""Bind qualified consumers to M29; add two independently checked channel families."""
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import MODES,pin,read,save
from consumer_checks import OLD,CHANNELS

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-convolution-four-numerics-amd-20260923'
PRODUCT=ROOT/'artifacts/pyannote-convolution-four-build-amd-20260923'
LOCAL=ROOT/'artifacts/pyannote-spatial-weight-reuse-20260922'
AMD=ROOT/'artifacts/pyannote-spatial-weight-reuse-amd-20260922'
PROBES=ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v2-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('channel_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
UPSTREAM=[(PRODUCT,'product','c0894aee340810bf0724ac52812fdeeb20318b8586f0c5e939584bcd667da0bb'),
    (LOCAL,'local','b6b1ce3bfe8fe85f390ed4bc4af03f0090c9a3f1226d7c2d2edfde684900adf5'),
    (AMD,'previous-amd','f7df7b7652a2dab086d2e364e2a1b52b472c471a0f1956bb40e5af4fa982b5aa')]

def previous_closed():
    for folder,label,digest in UPSTREAM:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    for name,wanted in read(LOCAL/'inputs.json')['files'].items():
        if '/bridge/' in name:assert pin(name)==wanted,name

def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    product=read(PRODUCT/'analysis.json')['built'];core=product['Lokad.Onnx.dll'];prior={}
    for mode in MODES:
        source_mode='raw' if mode in CHANNELS else mode
        folder=LOCAL/'consumers'/source_mode;target=bundle/'consumers'/mode
        for p in folder.iterdir():
            if p.is_file():copy(p,target/p.name)
        filename='ModelProbe.cs' if mode=='layers' else 'Probe.cs'
        before=(target/filename).read_text(encoding='utf8');assert before.count(OLD)==1
        after=before.replace(OLD,core['sha256'])
        if mode in CHANNELS:
            marker='foreach (int c in new[] { 16, 32 })';assert after.count(marker)==1
            left,right=CHANNELS[mode];after=after.replace(marker,f'foreach (int c in new[] {{ {left}, {right} }})')
        (target/filename).write_text(after,encoding='utf8',newline='\n')
        (bundle/(mode+'-consumer.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='qualified/'+filename,tofile=mode+'/'+filename)),encoding='utf8')
        assembly='LayerGraphs.dll' if mode=='layers' else 'Lokad.Onnx.Backend.Tests.dll'
        prior[mode]=pin(folder/'bin/Release/net10.0'/assembly)
        if mode not in CHANNELS:copy(LOCAL/'output'/(mode+'-256.json'),bundle/('windows-'+mode+'.json'))
    copy(ROOT/'global.json',bundle/'global.json')
    for suffix in ['dll','deps.json','runtimeconfig.json']:copy(PROBES/'source/bridge/bin/Release/net10.0'/('Bridge.'+suffix),bundle/'bridge'/('Bridge.'+suffix))
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(),str(p))
        if p.name in ['protocol.py','remote.py','remote_prepare.py','checks.py','consumer_checks.py']:copy(p,bundle/'tools'/p.name)
    copy(AMD/'payload/fixtures/result.json',bundle/'fixtures/result.json')
    for folder,label,digest in UPSTREAM:copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    save(bundle/'stage.json',dict(passed=True,core=core,product=product,prior_consumers=prior,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),core=core)))

if __name__=='__main__':prepare()
