"""Close the first actual-product Windows qualification without replaying calls."""
from pathlib import Path
import sys,subprocess,datetime,xml.etree.ElementTree as ET,shutil
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/pyannote/filterbank-precision'))
from shared import pin,read,write,absent,np,metric


def main():
    base=ROOT/'artifacts/wespeaker-frame-product-local-20260920';tests=ROOT/'artifacts/wespeaker-frame-tests-trx-20260920'
    proof=ROOT/'artifacts/wespeaker-precision-20260920';state=read(base/'run.json');recovery=read(tests/'run.json');spec=read(proof/'manifest.json');prior=read(proof/'audit.json')
    assert state['source']==recovery['source']=='1d10d22f73282bb00630371887ca3719d2e1553b'
    assert state['product']==recovery['product']==pin(ROOT/'src/Lokad.Onnx.Data/WeSpeakerAudio.cs')
    subprocess.run(['git','diff','--exit-code',state['source'],'--','src','tests/Lokad.Onnx.Backend.Tests/WeSpeakerAudioTests.cs','tests/pyannote/frame-product/Program.cs'],cwd=ROOT,check=True)
    assert pin(proof/'closed.json')['sha256']=='16aadd8e4e890c289a68f8bb0dccb738de1d87248396cc046f4b619177c05b0a'
    assert read(proof/'final-verification.json')['verified']
    births=[];resources=[]
    for folder,run_state,names in [(base,state,['test','consumer-build','old-dc','new-dc','corpus']),(tests,recovery,['test-trx'])]:
        assert run_state['complete'] and run_state['code']==0 and not run_state.get('error')
        assert [r['name'] for r in run_state['runs']]==names;births.append(run_state['supervisor'])
        for run in run_state['runs']:
            assert run['complete'] and run['code']==0 and not run.get('error') and 0<run['seconds']<600
            births.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
            assert run['members'][str(run['child']['pid'])]==run['child']['birth']
            rows=[__import__('json').loads(line) for line in (folder/(run['name']+'.samples.jsonl')).read_text().splitlines()];assert rows and len(rows)==run['samples'];previous=0
            for row in rows:
                assert previous<=row['seconds']<=run['seconds'] and row['seconds']-previous<10;previous=row['seconds']
                assert row['available']>=1024**3 and sum(m['rss'] for m in row['members'])<2*1024**3
                for m in row['members']:assert m['affinity']==[0] and m['rss']>=0 and run['members'][str(m['pid'])]==m['birth']
            assert run['seconds']-previous<10 and not (folder/(run['name']+'.stderr')).read_text().strip()
            resources.append(dict(name=run['name'],seconds=run['seconds'],samples=len(rows),peak_rss=max(sum(m['rss'] for m in row['members']) for row in rows)))
    assert all(absent(b) for b in births)
    tree=ET.parse(tests/'results/affected.trx').getroot();ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    counters=tree.find('t:ResultSummary/t:Counters',ns).attrib;results=tree.findall('t:Results/t:UnitTestResult',ns)
    assert counters['total']==counters['executed']==counters['passed']=='99' and counters['failed']=='0'
    assert len(results)==99 and all(r.attrib['outcome']=='Passed' for r in results)
    regression=[r.attrib for r in results if 'FrameMeanRemoval' in r.attrib['testName']];assert len(regression)==2
    new=read(base/'corpus/result.json');assert new['complete'] and new['phase']=='corpus' and len(new['rows'])==53
    assert [r['name'] for r in new['rows']]==[c['name'] for c in spec['cases']]
    checks=[]
    for case,row in zip(spec['cases'],new['rows']):
        p=base/'corpus'/(case['name']+'.f32');wanted=proof/'managed'/case['name']/'Frame/features.f32'
        assert pin(p)==row['output'] and pin(wanted)==prior['files'][wanted.relative_to(proof).as_posix()] and p.read_bytes()==wanted.read_bytes()
        assert row['input']==pin(proof/'inputs'/(case['name']+'.f32'))['sha256'] and row['shape']==case['shapes']['features']
        actual=np.fromfile(p,dtype='<f4').reshape(row['shape'])
        for engine in ['numpy','torch']:
            for coefficients,path in [('managed',proof/engine/case['name']/'features.npy'),('native',ROOT/case['old_reference']/engine/case['old_name']/'features.npy')]:
                expected=np.load(path,allow_pickle=False);value=metric(actual,expected,1e-4);assert value['failed']==0
                checks.append(dict(name=case['name'],engine=engine,coefficients=coefficients,**value))
    for name,raw in [('Window','window.f32'),('MelWeights','mel.f32')]:assert pin(base/'corpus'/(name+'.f32'))==new['tables'][name]==pin(proof/'inputs'/raw)
    phases={name:read(base/name/'result.json') for name in ['old-dc','new-dc','corpus']}
    for name,result in phases.items():
        runtime=result['runtime'];run=next(r for r in state['runs'] if r['name']==name)
        assert result['complete'] and runtime['pid']==run['child']['pid'] and runtime['affinity']==runtime['processor_count']==1 and runtime['framework']=='10.0.12'
        for row in result['loaded']:assert row['pin']==pin(row['file'])
    assert [r['failed'] for r in phases['old-dc']['rows']]==[6,256]
    assert all(r['failed']==0 and r['maximum']==0 for r in phases['new-dc']['rows'])
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Lokad.Onnx.Backend.Tests.dll']:
        assert pin(base/'product-bin'/name)==pin(ROOT/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'/name)
    # Capture historical source before changing working-tree-dependent validations.
    previous=subprocess.check_output(['git','show','a54db70:src/Lokad.Onnx.Data/WeSpeakerAudio.cs'],cwd=ROOT)
    with (base/'source-before.cs').open('xb') as f:f.write(previous)
    assert pin(base/'source-before.cs')['sha256']=='801be94d80e99d897b2067093dd6ad4514e6b9a5cd91d8cd50cf14a77bb77bf1'
    shutil.copyfile(ROOT/'src/Lokad.Onnx.Data/WeSpeakerAudio.cs',base/'source-after.cs')
    for path in [ROOT/'.agent/run-frame-product-local-20260920.py',ROOT/'.agent/run-frame-product-test-trx-20260920.py']:
        shutil.copyfile(path,base/path.name)
    copied=base/'test-trx-evidence';shutil.copytree(tests,copied)
    observation=dict(passed=True,source=state['source'],product=state['product'],test_counters=counters,regression_tests=regression,
        exact_arrays=53,values=3266560,checks=checks,dc={name:phases[name]['rows'] for name in ['old-dc','new-dc']},resources=resources,births=births,
        loaded=new['loaded'],assemblies={p.name:pin(p) for p in (base/'product-bin').iterdir()})
    folder=Path(__file__).resolve().parent;data=folder/'local-observations-20260920.json';report=folder/'local-results-20260920.md';write(data,observation)
    lines=['# Public WeSpeaker frame precision — Windows qualification, September20,2026','',
      '**The public frontend now preserves frame preprocessing in double.** Source `1d10d22` changes four declarations, retaining the coefficients, FFT, spectrum/log/centering arithmetic, float output and API contracts. All53 actual public outputs (3,266,560 values) match the proven Frame variant bit for bit and pass both independent managed-table and native-table double references at the unchanged1e-4 threshold. Maximum scaled error is1.99571e-6 against managed tables and5.82965e-5 against native tables.','',
      'A separate regression uses small exactly representable float signals and an exactly representable0.5 DC offset. Frame mean removal should eliminate that constant. The archived original DLL has6/160 and256/7,840 failing feature values for560/16,000 samples, with maxima0.003742218/0.045996666. The new public API produces identical features in both offset settings: zero differing error and zero failures. Both new unit cases pass.','',
      '**99 affected WeSpeaker/Community-1 tests pass**, including ownership, concurrency, cancellation, invalid inputs, pooling and clustering contracts. The first test command exited0 but its inner console summary was absent. That attempt is retained; a separate no-build/no-restore invocation adds a TRX logger, which independently records all99 cases and both new regression tests. No inference corpus was repeated for that logging repair.','',
      'The corpus consumer verifies complete shapes, immutable inputs, all held outputs after subsequent calls and exact coefficient tables. It loads the actual built Data/core assemblies through the public method; it does not execute the generated diagnostic variant. All source, binaries, raw outputs, command logs, resource samples and TRX records are retained. Every observed process birth is terminal. These are local working-tree builds on Windows/.NET10.0.12, CPU0; source-archive/AMD and connected natural-meeting qualification remain required.','',
      'Direct native float comparison still has187 failures for the new frontend, versus32 historically. This is not labelled passing native conformance. The [complete precision proof](../filterbank-precision/results-20260920.md) retains all comparisons and seven coordinates where native/reference1e-4 intervals are disjoint. The new mathematical frontend criterion is prospective; historical failures and all other model gates remain unchanged. No new application timing, DER or e5 result follows.','',
      '[Complete local observations](local-observations-20260920.json). Artifact `artifacts/wespeaker-frame-product-local-20260920`; the separate TRX recovery is retained there and at `artifacts/wespeaker-frame-tests-trx-20260920`.','']
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    receipt=dict(passed=True,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=state['source'],births=births,
        reports={str(p.relative_to(ROOT)).replace('\\','/'):pin(p) for p in [report,data,Path(__file__)]},
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',receipt);print(__import__('json').dumps(dict(receipt=pin(base/'closed.json'),files=len(receipt['files']),tests=99,arrays=53)))


if __name__=='__main__':main()
