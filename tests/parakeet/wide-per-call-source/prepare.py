"""Prepare a bounded wide-matrix dispatch change after current AMD attribution."""
import difflib,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-wide-per-call-source-20260923'
PROOF=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
PROFILE=ROOT/'artifacts/parakeet-current-profile-amd-20260923'
FILE='src/Lokad.Onnx/TensorOps.MatMul.cs'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def transform(source):
    start=source.index('    static unsafe void RunFloatMatMulKernel(')
    end=source.index('\n    public static ',start)
    before=source[start:end]
    old='!AblationSwitches.EnablePackedAvx512Dynamic || n < 64'
    new='!(AblationSwitches.EnablePackedAvx512Dynamic || (n >= 1024 && k >= 1024)) || n < 64'
    assert before.count(old)==2 and source.count(old)==2
    after=before.replace(old,new)
    assert after.replace(new,old)==before
    return source[:start]+after+source[end:]
def main():
    assert not BASE.exists()
    for folder,digest,relative in [(PROOF,'62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0',False),
        (PROFILE,'ea3e6471009edd07f0f20905c5e230f55461d7722126a32fabd227c3286d00ff',True)]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin((ROOT if relative else folder)/name)==wanted,name
    profile=read(PROFILE/'analysis.json');assert profile['passed'] and profile['calls']==240
    shares={}
    for d in profile['diagnostics']:
        cost=sum(r['seconds'] for r in d['exclusive'] if any(name in r['method'] for name in ['.mm_unsafe_vectorized_intrinsics_2x4packed_bump(','.mm_unsafe_vectorized_intrinsics_3x4packed(']))
        share=cost/d['selected_seconds']['corpus'];assert share>=.20
        shares[d['name']]=share
    assert set(shares)=={'sampled-a','sampled-b'}
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
    shutil.copy2(ROOT/'.agent/m39-parakeet-wide-per-call-20260923.md',BASE/'prospective-plan.md')
    value=dict(passed=True,built=False,numerically_qualified=False,root_product_changed=False,before=expected,changed=changed,
        permitted_compiled_change='Lokad.Onnx.Tensor<T>.RunFloatMatMulKernel',source={p.relative_to(source).as_posix():pin(p) for p in source.rglob('*') if p.is_file()},
        patch=pin(BASE/'candidate.patch'),plan=pin(BASE/'prospective-plan.md'),generator=pin(Path(__file__)),
        root_closure=pin(PROOF/'closed.json'),profile_closure=pin(PROFILE/'closed.json'),packed_consumer_shares=shares,
        scope='Only two existing per-call consumer guards allow n>=1024 and k>=1024; unchanged packing, kernels, buffers, cache caps, tails and explicit switch semantics.')
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),patch=value['patch'],changed=changed,shares=shares,root_product_changed=False)))
if __name__=='__main__':main()
