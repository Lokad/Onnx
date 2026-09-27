"""One fixed column-remainder consumer over the qualified root snapshot."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-pointwise-tail-source-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-lstm-layout-root-amd-20260927'
DIAGNOSIS = ROOT/'artifacts/parakeet-pointwise-cost-amd-20260927'
MATH = 'src/Lokad.Onnx/MathOps.cs'
HELPER = 'src/Lokad.Onnx/MathOps.PackedColumnTails.cs'


def pin(path):
    data=path.read_bytes()
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def write(path,value):
    with path.open('x',encoding='utf8') as stream: json.dump(value,stream,indent=2);stream.write('\n')


def change(raw):
    original=raw.decode().replace('\r\n','\n')
    start=original.index('    public unsafe static void mm_unsafe_vectorized_intrinsics_2x4packed_bump(')
    stop=original.index('    /// Register-tiled matrix multiplication reading panel-packed B, three',start)
    before=original[start:stop];after=before
    old='''            for (int tt = 0; tt < rv; tt++)
            {
                for (int i = 0; i < M; i += 2)'''
    new='''            int sharedRows = M >= 64 && N >= 64 ? M - M % 8 : 0;
            for (int tt = 0; tt < rv; tt++)
            {
                if (sharedRows > 0)
                    PackedColumnTailEightRows(sharedRows, N, K, rem, A,
                        T + tt * Vector256<float>.Count, C + blocked + tt * Vector256<float>.Count);
                for (int i = sharedRows; i < M; i += 2)'''
    pairs=[(old,new),
        ('''            int tail = rem - vcols;
            if (tail > 0)
            for (int i = 0; i < M; i += 2)''',
         '''            int tail = rem - vcols;
            int maskedRows = tail > 0 && Avx2.IsSupported ? sharedRows : 0;
            if (maskedRows > 0)
                PackedColumnMaskedEightRows(maskedRows, N, K, rem, tail, A,
                    T + vcols, C + blocked + vcols);
            if (tail > 0)
            for (int i = maskedRows; i < M; i += 2)''')]
    for old,new in pairs:
        assert after.count(old)==1;after=after.replace(old,new)
    restored=after
    for old,new in reversed(pairs):
        assert restored.count(new)==1;restored=restored.replace(new,old)
    assert restored==before
    assert after.split('        int rem = K - blocked;')[0]==before.split('        int rem = K - blocked;')[0]
    value=original[:start]+after+original[stop:]
    return (value.replace('\n','\r\n') if b'\r\n' in raw else value).encode(),list(difflib.unified_diff(
        before.splitlines(True),after.splitlines(True),fromfile=MATH,tofile=MATH))


def main():
    assert not BASE.exists()
    assert pin(QUALIFIED/'closed.json')['sha256']=='efb99eea455647c64bbe0811c1b7d46387add59049ac81e48a926b250e7c42da'
    assert pin(DIAGNOSIS/'closed.json')['sha256']=='4cd3c5c3ffb4c71c40483544024426b655556b14a33e0c562304a84800e705f0'
    assert read(DIAGNOSIS/'closed.json')['usable_for_candidate_selection']
    closure=read(QUALIFIED/'closed.json')
    assert closure['passed'] and closure['files']['bundle/stage.json']==pin(QUALIFIED/'bundle/stage.json')
    stage=read(QUALIFIED/'bundle/stage.json')
    source={n.removeprefix('source/'):v for n,v in stage['files'].items() if n.startswith('source/')}
    assert len(source)==443
    values={}
    for name,wanted in source.items():
        path=QUALIFIED/'bundle/source'/name
        assert pin(path)==wanted and pin(ROOT/name)==wanted,name
        values[name]=path.read_bytes()
    values[MATH],patch=change(values[MATH])
    assert HELPER not in values;values[HELPER]=(TOOLS/'MathOps.PackedColumnTails.cs.txt').read_bytes()
    BASE.mkdir()
    for name,data in values.items():
        path=BASE/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    (BASE/'candidate.patch').write_text(''.join(patch),encoding='utf8')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'PLAN.md').read_bytes())
    write(BASE/'prepared.json',dict(passed=True,release_admitted=False,root_product_changed=False,
        baseline=pin(QUALIFIED/'closed.json'),diagnosis=pin(DIAGNOSIS/'closed.json'),
        source={n:pin(BASE/'source'/n) for n in values},source_before=source,
        changed_methods=['mm_unsafe_vectorized_intrinsics_2x4packed_bump'],added_methods=['PackedColumnTailEightRows','PackedColumnMaskedEightRows'],
        full_panel_source_unchanged=True,patch=pin(BASE/'candidate.patch'),plan=pin(BASE/'prospective-plan.md'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),files=len(values))))


if __name__=='__main__':main()
