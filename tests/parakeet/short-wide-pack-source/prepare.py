"""Prepare a bounded wide-matrix dispatch change after current AMD attribution."""
import difflib,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-short-wide-pack-source-20260923'
PROOF=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
SCREEN=ROOT/'artifacts/parakeet-wide-per-call-screen-amd-20260923'
FILE='src/Lokad.Onnx/TensorOps.MatMul.cs'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def transform(source):
    start=source.index('    static unsafe void RunFloatMatMulKernel(')
    end=source.index('\n    public static ',start)
    before=source[start:end]
    old='        const int TiledPackMinRows = 64;'
    new=('        // Short, wide projections reuse the same AVX2 panel consumers.\n'
         '        // Keep the original threshold for smaller axes and every other route.\n'
         '        int TiledPackMinRows = n >= 1024 && k >= 1024 ? 48 : 64;')
    assert before.count(old)==1 and before.count('TiledPackMinRows')==4
    after=before.replace(old,new)
    assert after.replace(new,old)==before
    return source[:start]+after+source[end:]
def main():
    assert not BASE.exists()
    for folder,digest,relative in [(PROOF,'62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0',False),
        (SCREEN,'66a3f377871e962d659d37f0b1f878107028ca31c23121d2466ffd0f0efecee7',False)]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin((ROOT if relative else folder)/name)==wanted,name
    screen=read(SCREEN/'analysis.json');assert not screen['admitted'] and all(c['passed'] for c in screen['controls'])
    assert screen['rows'][0]['current']['value']>screen['rows'][3]['current']['value']
    assert screen['rows'][1]['current']['value']>screen['rows'][4]['current']['value']
    trace=ROOT/'artifacts/parakeet-performance-profile-v2-20260921'
    trace_proof=read(trace/'closed.json');assert trace_proof['passed']
    assert pin(trace/'closed.json')['sha256']=='6d8ce878f99acf291cb20348bba948bd1de7f128298c774cf94adf84e855700d'
    shape_sources={};frames={}
    for file in sorted((trace/'trace-output').glob('*.json')):
        value=read(file)
        if value.get('graph')!='encoder' or value.get('phase')!='unprofiled':continue
        name=file.relative_to(ROOT).as_posix();assert pin(file)==trace_proof['files'][name]
        shape=value['outputs']['outputs']['shape'];assert shape[:2]==[1,1024]
        frames.setdefault(value['name'],shape[2]);assert frames[value['name']]==shape[2]
        shape_sources[name]=pin(file)
    assert len(frames)==20 and sorted(n for n in frames.values() if n<64)==[51,61]
    baseline=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
    assert pin(baseline/'closed.json')['sha256']=='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'
    assert pin(baseline/'analysis.json')==read(baseline/'closed.json')['files']['analysis.json']
    timing={r['name']:r for r in read(baseline/'analysis.json')['table'] if not r['is_corpus']}
    assert set(timing)==set(frames)
    census=[dict(name=name,frames=m,current_seconds=timing[name]['current']['seconds'],ort_seconds=timing[name]['ort']['seconds']) for name,m in frames.items()]
    expected={name.removeprefix('source/'):wanted for name,wanted in read(PROOF/'bundle/stage.json')['files'].items() if name.startswith('source/')}
    assert len(expected)==420
    for name,wanted in expected.items():assert pin(ROOT/name)==wanted,name
    before=(ROOT/FILE).read_text(encoding='utf8');after=transform(before)
    BASE.mkdir();source=BASE/'source';source.mkdir()
    for name in expected:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    (source/FILE).write_text(after,encoding='utf8',newline='\n')
    changed=[name for name,wanted in expected.items() if pin(source/name)!=wanted];assert changed==[FILE]
    for name,wanted in expected.items():assert pin(ROOT/name)==wanted,name
    (BASE/'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=FILE,tofile=FILE)),encoding='utf8')
    shutil.copy2(ROOT/'.agent/m40-parakeet-short-wide-pack-20260923.md',BASE/'prospective-plan.md')
    (BASE/'census.json').write_text(json.dumps(dict(passed=True,rows=census,shape_sources=shape_sources,trace_closure=pin(trace/'closed.json'),baseline_closure=pin(baseline/'closed.json')),indent=2)+'\n',encoding='utf8')
    value=dict(passed=True,built=False,numerically_qualified=False,root_product_changed=False,before=expected,changed=changed,
        permitted_compiled_change='Lokad.Onnx.Tensor<T>.RunFloatMatMulKernel',source={p.relative_to(source).as_posix():pin(p) for p in source.rglob('*') if p.is_file()},
        patch=pin(BASE/'candidate.patch'),plan=pin(BASE/'prospective-plan.md'),generator=pin(Path(__file__)),
        root_closure=pin(PROOF/'closed.json'),screen_closure=pin(SCREEN/'closed.json'),census=pin(BASE/'census.json'),
        scope='Only local minimum packed rows changes from64to48when both axes>=1024; allthreeexistinguses share it, including exact-three-row suppression; no M39 AVX512 change, arithmetic, budgets or retained storage changes.')
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),patch=value['patch'],changed=changed,short_clips=[r for r in census if r["frames"]<64],root_product_changed=False)))
if __name__=='__main__':main()
