"""Create the source-policy successor; retain the original failed campaign intact."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'tests/parakeet/owned-batch-isolation-root-amd'
NEW = ROOT/'tests/parakeet/owned-batch-isolation-root-policy-amd'


def replace(text, old, new):
    assert text.count(old)==1, old
    return text.replace(old,new)


def main():
    assert not NEW.exists(); NEW.mkdir()
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','new_cases.py','graph_prerequisite.py',
                 'admission.py','audit.py','warning_census.py','test_guards.py','test_census.py','prepare.py','run.py']:
        text=(OLD/name).read_text(encoding='utf8')
        if name in ['run.py','prepare.py']:
            text=text.replace('owned-batch-isolation-root-amd-20260925','owned-batch-isolation-root-policy-amd-20260925')
            text=text.replace("'/dev/shm/lokad-parakeet-owned-batch-isolation-root-20260925'", "'/dev/shm/lokad-parakeet-owned-batch-isolation-root-policy-20260925'")
        if name=='protocol.py':
            text=replace(text,"'tensors-build','inventory'","'tensors-build','bridge-restore','bridge-build','inventory'")
        elif name=='remote.py':
            text=replace(text,"'consumer':BASE/'consumer/PackageProbe.csproj'","'bridge':BASE/'bridge-source/Bridge.csproj',\n          'consumer':BASE/'consumer/PackageProbe.csproj'")
            text=replace(text,"BASE/'bridge/Bridge.dll',BASE/'measured',BASE/'runtime',BASE/name/'instructions.json'",
                "BASE/'built/Bridge.dll',BASE/'measured',BASE/'runtime',BASE/name/'instructions.json',BASE/'measured'")
            text=replace(text,"    if name=='inventory':\n",'''    if name=='bridge-build':
        folder=BASE/'bridge-source/bin/Release/net10.0'
        for suffix in ['dll','deps.json','runtimeconfig.json']:shutil.copy2(folder/('Bridge.'+suffix),BASE/'built'/('Bridge.'+suffix))
        save(BASE/'bridge-built.json',dict(passed=True,source=pin(BASE/'bridge-source/Program.cs'),
            files={p.name:pin(p) for p in (BASE/'built').glob('Bridge.*')}))
    if name=='inventory':
''')
        elif name=='prepare.py':
            text=replace(text,"APPLIED=ROOT/'artifacts/parakeet-owned-batch-isolation-root-integration-20260925'",
                "APPLIED=ROOT/'artifacts/parakeet-owned-batch-isolation-root-policy-20260925/integration'")
            text=replace(text,"    for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']:copy(SELECTED/'bundle/bridge'/name,bundle/'bridge'/name)",
                "    copy(ROOT/'tests/parakeet/owned-batch-isolation-build/Bridge.cs.txt',bundle/'bridge-source/Program.cs')\n    copy(ROOT/'tests/parakeet/selected-profile-build-amd/Bridge.csproj',bundle/'bridge-source/Bridge.csproj')\n    copy(ROOT/'global.json',bundle/'bridge-source/global.json')")
            text=replace(text,"copy(APPLIED/'applied.json',bundle/'evidence/root-applied.json')",
                "copy(APPLIED/'applied.json',bundle/'evidence/root-applied.json')\n    copy(APPLIED.parent/'applied.json',bundle/'evidence/policy-correction.json')")
            text=text.replace('435 actual root inputs: exact measured relocation product and qualified tests with portable hardware guards. Require all3281 Core/697 Data bodies, flags and public declarations equal measured binaries.',
                '435 actual root inputs. All3281 Core/697 Data method bodies and implementation flags match measured binaries. Only PrepareOwnedMatMulWeights becomes public and Data friendship is removed; helpers use explicit original arguments.')
        elif name=='checks.py':
            start=text.index('def inventory('); end=text.index('\n\ndef census(',start)
            text=text[:start]+'''def metadata_delta(row):
    public_before=set(row['public_surface']);public_after=set(row['public_surface_after'])
    attributes_before=set(row['assembly_attributes_before']);attributes_after=set(row['assembly_attributes_after'])
    assert len(public_before)==len(row['public_surface']) and len(public_after)==len(row['public_surface_after'])
    if row['assembly']=='Lokad.Onnx.dll':
        added={'MEMBER Lokad.Onnx.ComputationalGraph Method Int32 PrepareOwnedMatMulWeights()',
               'MEMBER Lokad.Onnx.ComputationalGraph Method Int32 PrepareOwnedMatMulWeights() FLAGS Public, HideBySig'}
        friend='[System.Runtime.CompilerServices.InternalsVisibleToAttribute("Lokad.Onnx.Data")]'
        assert not row['public_surface_equal'] and public_after-public_before==added and not public_before-public_after
        assert attributes_before-attributes_after=={friend} and not attributes_after-attributes_before
    else:
        assert row['assembly']=='Lokad.Onnx.Data.dll' and row['public_surface_equal']
        assert public_before==public_after and attributes_before==attributes_after
    return True


def inventory(value,measured,built):
    assert value['inventory_complete'] and len(value['observations'])==2
    for row,(name,count) in zip(value['observations'],[('Lokad.Onnx.dll',3281),('Lokad.Onnx.Data.dll',697)]):
        assert row['assembly']==name and row['methods']==row['unchanged_methods']==count
        assert metadata_delta(row)
        assert not row['differences'] and not row['added'] and not row['removed']
        assert row['before_sha256']==measured[name]['sha256'] and row['after_sha256']==built[name]['sha256']
        assert len(row['normalized_methods'])==count and not row['candidate_methods']
        assert row['method_flags_before']==row['method_flags_after'] and len(row['method_flags_after'])==count
    core=value['observations'][0]
    key,=[k for k in core['normalized_methods'] if '::RunBatchedFloatMatMul::' in k]
    assert value['release']['sha256']==measured['Lokad.Onnx.dll']['sha256']
    assert value['release']['methods']=={key:core['normalized_methods'][key]}
    return dict(passed=True,core_methods=3281,data_methods=697,public_surface_equal=False,
        public_surface_delta_exact=True,data_friend_removed=True,method_bodies_equal=True,implementation_flags_equal=True)
''' + text[end:]
        elif name=='audit.py':
            text=replace(text,"    built=read(collected/'built.json');assert built['passed']",'''    bridge=read(collected/'bridge-built.json');assert bridge['passed']
    assert bridge['source']==payload['files']['bridge-source/Program.cs']
    assert set(bridge['files'])=={'Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json'}
    for name,wanted in bridge['files'].items():assert pin(collected/'built'/name)==wanted
    built=read(collected/'built.json');assert built['passed']''')
        elif name=='warning_census.py':
            text=replace(text,"        if path.suffix not in ['.stdout','.stderr']:continue", "        if path.suffix not in ['.stdout','.stderr'] or path.name.startswith(('bridge-restore.','bridge-build.')):continue")
            text=replace(text,"    before,after=census(reference),census(folder)",'''    for path in (folder/'logs').iterdir():
        if path.name.startswith(('bridge-restore.','bridge-build.')):
            assert path.suffix in ['.stdout','.stderr'] or path.suffix=='.jsonl'
            if path.suffix in ['.stdout','.stderr']:
                warnings=[line.strip() for line in path.read_text(encoding='utf8').splitlines() if re.search(r'\\bwarning\\b',line,re.IGNORECASE)]
                assert all(line=='0 Warning(s)' for line in warnings),warnings
    before,after=census(reference),census(folder)''')
        if text==(OLD/name).read_text(encoding='utf8'):
            shutil.copyfile(OLD/name,NEW/name)
        else:
            (NEW/name).write_text(text,encoding='utf8')
    print(NEW)


if __name__=='__main__':main()
