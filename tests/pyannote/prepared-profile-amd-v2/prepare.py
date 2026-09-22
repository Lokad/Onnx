"""Bind the existing Linux diagnostic consumer to the newly selected product."""
import ast
import difflib
import shutil
import tarfile
import traceback
from common import *

MODEL = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922/runtime'
NATIVE = ROOT/'artifacts/pyannote-input-address-screen-20260922'


def main():
    assert not BASE.exists()
    failed = ROOT/'artifacts/pyannote-prepared-profile-amd-20260922'
    failure = read(failed/'failure-closed.json')
    assert failure['retained_failure'] and not failure['passed'] and failure['normalized_source_equal']
    for name, wanted in failure['files'].items(): assert pin(failed/name) == wanted, name
    verify(failure['tools'])

    prior = read(PRIOR/'closed.json'); assert prior['passed']; verify(prior['files'])
    assert pin(PRIOR/'closed.json')['sha256'] == 'cc9b31a0cb9abe795c24789731ff462515b394133058f4ea6a62b30fd3f3b2cc'
    for identity in prior['local_identities']: terminal(identity)
    native = read(NATIVE/'closed.json'); assert native['passed']
    assert pin(NATIVE/'closed.json')['sha256'] == '8d52d292ce04fa6d80f0e91ea8c476a25982009176af3fd5009d4ff9237694d9'
    for name,wanted in native['files'].items(): assert pin(NATIVE/name) == wanted,name
    assert not read(NATIVE/'analysis.json')['admitted']
    amd = read(AMD/'closed.json'); assert amd['passed']
    assert pin(AMD/'closed.json')['sha256'] == '5c238cd33845eb58fc00332361a967185ae82a9fcb70854530e07fad58f064d0'
    for name,wanted in amd['files'].items(): assert pin(AMD/name) == wanted,name
    original = read(PRIOR/'payload/payload.json')
    for name,wanted in original['files'].items(): assert pin(PRIOR/'payload'/name) == wanted,name
    BASE.mkdir(); (BASE/'logs').mkdir(); payload = BASE/'payload'
    shutil.copytree(PRIOR/'payload',payload)
    (payload/'payload.json').unlink()  # This is a new unfrozen local copy.
    files = {rel(p):pin(p) for p in [failed/'failure-closed.json',PRIOR/'closed.json',NATIVE/'closed.json',AMD/'closed.json', ROOT/'artifacts/pyannote-selected-profile-amd-20260922/payload/tools/stacks.py']}
    for name,sha in [('Lokad.Onnx.dll',CORE),('Lokad.Onnx.Data.dll',DATA)]:
        source = MODEL/name; assert pin(source)['sha256'] == sha
        shutil.copy2(source,payload/'runtime'/name); files[rel(source)] = pin(source)
    source = BASE/'consumer'; source.mkdir()
    for name in ['Program.cs','Diagnostic.cs','NpySupport.cs','SampledAudio.csproj']:
        shutil.copy2(PRIOR/'consumer'/name,source/name); files[rel(PRIOR/'consumer'/name)] = pin(PRIOR/'consumer'/name)
    program = source/'Program.cs'; before = program.read_text(encoding='utf8'); after = before
    for old,new in [('e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838',CORE),
                    ('85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a',DATA)]:
        assert after.count(old) == 1; after = after.replace(old,new)
    program.write_text(after,encoding='utf8')
    save(BASE/'consumer-adaptation.json',dict(reason='Only expected selected Core/Data hash literals; Linux diagnostic and public checks unchanged',
        diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True)))))
    bridge = BASE/'bridge'; bridge.mkdir()
    for name in ['Program.cs','Bridge.csproj']: shutil.copy2(PRIOR/'bridge'/name,bridge/name)
    # Keep all previous request assets, thread markers, collector and accounting
    # bytes. Replace only product identities and the exact contemporary outputs.
    manifest_source = PAYLOAD/'manifests/portable-pyannote.json'
    manifest = read(manifest_source); assert manifest['core_sha256'] == CORE and manifest['data_sha256'] == DATA
    archive = read(PAYLOAD/'payload.json'); assert pin(manifest_source) == archive['files']['manifests/portable-pyannote.json']
    old_manifest = read(payload/'manifest.json')
    assert {k:v for k,v in old_manifest.items() if k not in ['core_sha256','data_sha256']} == {k:v for k,v in manifest.items() if k not in ['core_sha256','data_sha256']}
    shutil.copy2(manifest_source,payload/'manifest.json'); files[rel(manifest_source)] = pin(manifest_source)
    expected = AMD/'collected/campaign/timing-01-portable-output/result.json'
    assert pin(expected) == amd['files'][expected.relative_to(AMD).as_posix()]
    assert read(expected)['core_sha256'] == CORE and read(expected)['data_sha256'] == DATA
    shutil.copy2(expected,payload/'prior-amd-result.json'); files[rel(expected)] = pin(expected)
    assert pin(TOOLS/'remote.py') == pin(payload/'tools/remote.py')
    shutil.copy2(TOOLS/'README.md', payload/'prospective-current-plan.md')
    for folder in [source,bridge,TOOLS]:
        for p in folder.iterdir():
            if p.is_file():
                if p.suffix == '.py': ast.parse(p.read_text(),str(p))
                files[rel(p)] = pin(p)
    save(BASE/'inputs.json',dict(passed=True,files=files))
    state = new_state(); flags = monitor.FLAGS+['-p:NuGetAudit=false']
    def run(name,args,output):
        monitor.worker(state,BASE/'preparation.json',name,args,ROOT,[0],8,8,900,True,output)
        print(name,'passed',flush=True)
    try:
        for name,project,extra in [('consumer',source/'SampledAudio.csproj',['-p:FrozenProductDirectory='+str(payload/'runtime')]),
                                  ('bridge',bridge/'Bridge.csproj',[])]:
            run(name+'-restore',['dotnet','restore',project,*flags,'--source',FEED,'--packages',BASE/'packages',*extra],None)
            run(name+'-build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers',*extra],project.parent)
        built = source/'bin/Release/net10.0'
        run('consumer-instructions',['dotnet',bridge/'bin/Release/net10.0/Bridge.dll',PRIOR/'payload/runtime',built,BASE/'instructions.json'],bridge)
        inventory = read(BASE/'instructions.json'); assert inventory['inventory_complete']
        row = inventory['observations'][0]
        assert row['assembly'] == 'SampledAudio.dll' and row['methods'] == 160 and row['unchanged_methods'] == 159
        assert row['public_surface_equal'] and not row['added'] and not row['removed']
        assert len(row['differences']) == 1 and row['differences'][0].startswith('Program::<Main>$::')
        key, = row['differences']
        expected = row['normalized_methods'][key]
        for old, new in [('e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838', CORE),
                         ('85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a', DATA)]:
            assert expected.count(old) == 1
            expected = expected.replace(old, new)
        assert expected == row['candidate_methods'][key]
        for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']: assert pin(built/name) == pin(payload/'runtime'/name)
        for suffix in ['dll','deps.json','runtimeconfig.json']: shutil.copy2(built/('SampledAudio.'+suffix),payload/'runtime'/('SampledAudio.'+suffix))
        verify(files)
        spec = dict(original,core=CORE,data=DATA,consumer=pin(payload/'runtime/SampledAudio.dll'),
            files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()})
        save(payload/'payload.json',spec)
        with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
            for p in sorted(payload.rglob('*')):
                if p.is_file(): tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
        files.update({rel(p):pin(p) for p in payload.rglob('*') if p.is_file()})
        save(BASE/'prepared.json',dict(passed=True,files=files,archive=pin(BASE/'payload.tar.gz'),payload=pin(payload/'payload.json'),
            instructions=pin(BASE/'instructions.json'),existing_methods=160,unchanged_methods=159))
        state['code'] = 0
    except BaseException:
        state.update(code=1,error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE/'preparation.json',state)


if __name__ == '__main__': main()
