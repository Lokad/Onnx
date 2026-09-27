"""Partition retained decoder clocks and inspect the exact source; no execution."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-decoder-current-review-20260927'
REVISION = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
SOURCES = ['onnxruntime/core/providers/cpu/math/matmul.cc',
           'onnxruntime/core/providers/cpu/math/gemm.cc',
           'onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc']


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def analyze():
    inputs = {}; reports = []
    for folder,digest in [
        ('parakeet-pad-current-profile-resume-amd-20260927','5e71163ac5e84c6d5c0b688cf175160e6d80047614b0b162afc4c515c98f06d9'),
        ('parakeet-ort-diagnosis-amd-20260924','615353b4d8524f8935076357c0949859553c27826d50d53ebd82fb002d146a85')]:
        base = ROOT/'artifacts'/folder
        assert pin(base/'closed.json')['sha256'] == digest
        proof = read(base/'closed.json'); assert proof['passed'] and proof['analysis'] == pin(base/'analysis.json')
        inputs[folder] = dict(closure=pin(base/'closed.json'),analysis=pin(base/'analysis.json'))
        reports.append(read(base/'analysis.json'))
    managed,native = reports
    ours = {r['name']:r for r in managed['phases']['wall']['node_rows'] if r['graph']=='decoder'}
    theirs = {r['name']:r for r in native['profiles']['decoder']['node_clocks']}
    assert len(ours)==29 and len(theirs)==23
    assert all(r['calls']==3600 for r in [*ours.values(),*theirs.values()])
    rows=[]; used=set(); native_used=set()
    def group(label,a,b):
        assert len(a)==len(set(a)) and len(b)==len(set(b))
        assert not used.intersection(a) and not native_used.intersection(b)
        used.update(a); native_used.update(b)
        x=sum(ours[n]['corpus_seconds'] for n in a)
        y=sum(theirs[n]['exclusive_us'] for n in b)/3e6
        rows.append(dict(group=label,managed_seconds=x,ort_seconds=y,difference_seconds=x-y,
                         managed_members=a,ort_members=b))
    recurrent=['/decoder/dec_rnn/lstm/LSTM','/decoder/dec_rnn/lstm/LSTM_1']
    final=['/joint/joint_net/joint_net.2/MatMul','/joint/joint_net/joint_net.2/Add']
    pred=['/joint/pred/MatMul','/joint/pred/Add','/joint/Transpose_1']
    group('Both recurrent nodes',recurrent,recurrent)
    group('Final output projection and bias',final,final)
    group('Prediction projection and transpose',pred,pred)
    group('Encoder projection and transpose',
          ['/joint/enc/MatMul','/joint/enc/Add','/joint/Transpose'],
          ['/joint/enc/MatMul/MatmulTransposeFusion/','/joint/enc/Add'])
    group('All other decoder operators',sorted(set(ours)-used),sorted(set(theirs)-native_used))
    assert used==set(ours) and native_used==set(theirs)
    a=managed['phases']['wall']['phase_seconds']['decoder']
    b=native['phases']['profile']['corpus_phase_seconds']['decoder']
    x=a-sum(r['managed_seconds'] for r in rows); y=b-sum(r['ort_seconds'] for r in rows)
    assert min(x,y)>=0
    rows.append(dict(group='Decoder outside timed operators',managed_seconds=x,ort_seconds=y,difference_seconds=x-y))
    assert math.isclose(sum(r['managed_seconds'] for r in rows),a,abs_tol=1e-12)
    assert math.isclose(sum(r['ort_seconds'] for r in rows),b,abs_tol=1e-12)
    shapes=[r for r in native['profiles']['decoder']['shapes'] if r['name'] in recurrent+[final[0]]]
    assert len(shapes)==3 and all(r['calls']==4800 for r in shapes)
    projection=next(r for r in shapes if r['name']==final[0])
    assert projection['inputs']==[{'float':[1,1,1,640]}]
    assert projection['outputs']==[{'float':[1,1,1,8198]}]
    assert ours[final[0]]['constant_inputs'][1]['dims']==[640,8198]
    source_bytes={}
    for name in SOURCES:
        result=subprocess.run(['git','-c','gc.auto=0','-C',str(ROOT/'external/onnxruntime'),
                               'show',REVISION+':'+name],check=True,capture_output=True,timeout=30)
        source_bytes[name]=result.stdout
    for name in ['TensorOps.MatMul.cs','TensorOps.OwnedPackedMatMul.cs','Zzz.IsolatedShortMatMul.cs',
                 'MathOps.cs','CPUExecutionProvider.LstmPanels.cs','CPUExecutionProvider.Recurrent.cs']:
        path=ROOT/'src/Lokad.Onnx'/name;inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    text=(ROOT/'src/Lokad.Onnx/TensorOps.MatMul.cs').read_text()
    assert 'if ((m & 1) != 0 && (m % 3) != 0) return null;' in text
    assert 'const int M1BlockedMinColumns = 8192;' in text and 'mm_m1_kblocked(m, n, k, x, y, output);' in text
    native_text=source_bytes[SOURCES[0]].decode()
    assert 'const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(1);' in native_text
    assert 'data[i].BIsPacked = bool(packed_b_);' in native_text and 'MlasGemmBatch(' in native_text
    return dict(passed=True,new_inference_calls=0,product_changed=False,inputs=inputs,
        source_revision=REVISION,native_sources={n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in source_bytes.items()},
        partition=rows,managed_nodes=list(ours.values()),ort_nodes=list(theirs.values()),observed_native_shapes=shapes,
        managed_corpus_seconds=managed['corpus']['wall'],managed_decoder_seconds=a,ort_decoder_seconds=b,
        current_managed_dispatch_is_source_inferred=True,native_packed_buffer_dump=False,
        native_per_node_leaf_join=False,overhead_subtracted=False,native_profile_date='2026-09-24',
        selected_candidate=None),source_bytes


def main():
    assert sys.argv[1:] in [[],['--publish']]
    value,sources=analyze()
    target=OUT/'observations-20260927.json'
    if not sys.argv[1:]:
        assert read(target)==value
        print(json.dumps(dict(passed=True,partition=value['partition'])));return
    assert not BASE.exists() and not target.exists()
    BASE.mkdir();(BASE/'native-source').mkdir()
    for name,data in sources.items():(BASE/'native-source'/Path(name).name).write_bytes(data)
    raw=json.dumps(value,indent=2,allow_nan=False)+'\n'
    (BASE/'analysis.json').write_text(raw,encoding='utf8');target.write_text(raw,encoding='utf8')
    proof=dict(passed=True,analysis=pin(BASE/'analysis.json'),analyst=pin(Path(__file__)),inputs=value['inputs'],
               native_sources={p.name:pin(p) for p in (BASE/'native-source').iterdir()},new_inference_calls=0)
    (BASE/'closed.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,closure=pin(BASE/'closed.json'),partition=value['partition'])))


if __name__ == '__main__': main()
