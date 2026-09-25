"""Prepare observation calls over exact M78; no changed arithmetic or dispatch."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-depthwise-route-source-20260925'
PRIOR=ROOT/'artifacts/parakeet-packed-final-row-source-20260925'
CONV='src/Lokad.Onnx/TensorOps.ConvPool.cs';MATMUL='src/Lokad.Onnx/TensorOps.MatMul.cs'
HELPER='src/Lokad.Onnx/DepthwiseRouteProbe.cs'


def pin(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def read(path):return json.loads(path.read_text(encoding='utf8'))
def write(path,value):
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def edits(name):
    if name==CONV:
        return [
            ('        bool hasBias = bd is not null;\n        if (TryConvBlockedSpatial',
             '        bool hasBias = bd is not null;\n        DepthwiseRouteProbe.Enter(input, weight, xd, wd, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad.top, pad.left, pad.bottom, pad.right, outH, outW, options);\n        if (TryConvBlockedSpatial'),
            ('        var patchMem = new Memory<float>(scratch, 0, tileKFull * blockN);',
             '        DepthwiseRouteProbe.BeginTiled(b, group, C, H, W, M, kH, kW, outH, outW, blockN, scratch.Length);\n        var patchMem = new Memory<float>(scratch, 0, tileKFull * blockN);'),
            ('            int colCount = Math.Min(blockN, tileN - colStart);\n            unsafe',
             '            int colCount = Math.Min(blockN, tileN - colStart);\n            DepthwiseRouteProbe.Panel(colCount, tileKFull);\n            unsafe'),
            ('                int outBase = b * outBatch + g * tileM * tileN;',
             '                DepthwiseRouteProbe.Views();\n                int outBase = b * outBatch + g * tileM * tileN;'),
            ('                    tileM, tileKg, colCount, options))\n                    Tensor<float>.MatMul2D(wView, pView, dView, options);',
             '                    tileM, tileKg, colCount, options))\n                {\n                    DepthwiseRouteProbe.Product(tileM, tileKg, colCount);\n                    Tensor<float>.MatMul2D(wView, pView, dView, options);\n                }'),
            ('        }\n    }\n\n    /// <summary>\n    /// Runs 1x1 stride-1 no-pad batches',
             '        }\n        DepthwiseRouteProbe.EndTiled();\n    }\n\n    /// <summary>\n    /// Runs 1x1 stride-1 no-pad batches')]
    assert name==MATMUL
    return [
        ('        else if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported)\n        {\n            mm_unsafe_vectorized_intrinsics(m, n, k, x, y, output);',
         '        else if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported)\n        {\n            DepthwiseRouteProbe.Kernel("one-row-fma", m, n, k);\n            mm_unsafe_vectorized_intrinsics(m, n, k, x, y, output);'),
        ('        else if (options.UseSimd)\n        {\n            mm_unsafe_vectorized(m, n, k, x, y, output);',
         '        else if (options.UseSimd)\n        {\n            DepthwiseRouteProbe.Kernel("portable", m, n, k);\n            mm_unsafe_vectorized(m, n, k, x, y, output);'),
        ('        else\n        {\n            mm(m, n, k, x, y, output);',
         '        else\n        {\n            DepthwiseRouteProbe.Kernel("scalar", m, n, k);\n            mm(m, n, k, x, y, output);')]


def changed(name,original):
    pairs=edits(name);value=original
    for before,after in pairs:
        assert value.count(before)==1,(name,before);value=value.replace(before,after)
    restored=value
    for before,after in reversed(pairs):
        assert restored.count(after)==1;restored=restored.replace(after,before)
    assert restored==original
    return value


def main():
    assert not BASE.exists()
    assert pin(PRIOR/'prepared.json')['sha256']=='e3ca64b50ea4a5dd276d90a19779d602f8ed78b6275b9e5448fe3b021bec5e1b'
    previous=read(PRIOR/'prepared.json');assert len(previous['source'])==433
    values={}
    for name,wanted in previous['source'].items():
        path=PRIOR/'source'/name;assert pin(path)==wanted,name;values[name]=path.read_bytes()
    diagnosis=ROOT/'artifacts/parakeet-stem-diagnosis-20260925'
    assert pin(diagnosis/'closed.json')['sha256']=='469e436bf873876ff547db99ec1f42f2f6afd4da7669e83f911f039848944ca9'
    assert read(diagnosis/'closed.json')['analysis']==pin(diagnosis/'analysis.json')
    patch=[]
    for name in [CONV,MATMUL]:
        raw=values[name].decode();before=raw.replace('\r\n','\n');after=changed(name,before)
        values[name]=(after.replace('\n','\r\n') if '\r\n' in raw else after).encode()
        patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=name,tofile=name))
    assert HELPER not in values;values[HELPER]=(TOOLS/'DepthwiseRouteProbe.cs.txt').read_bytes()
    BASE.mkdir()
    for name,data in values.items():
        path=BASE/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    (BASE/'observer.patch').write_text(''.join(patch),encoding='utf8')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'PLAN.md').read_bytes())
    source={n:pin(BASE/'source'/n) for n in values}
    assert set(n for n,w in previous['source'].items() if source[n]!=w)=={CONV,MATMUL}
    write(BASE/'prepared.json',dict(passed=True,diagnostic_only=True,root_product_changed=False,release_admitted=False,
        baseline=pin(PRIOR/'prepared.json'),diagnosis=pin(diagnosis/'closed.json'),source=source,
        changed_product_files=[CONV,MATMUL],added_observer=HELPER,
        changed_methods=['Conv2DFloatCore','RunTiledBatchFloat','RunFloatMatMulKernel'],
        plan=pin(BASE/'prospective-plan.md'),patch=pin(BASE/'observer.patch'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(source=pin(BASE/'prepared.json'),files=len(source),changed_product_files=[CONV,MATMUL])))


if __name__=='__main__':main()
