from pathlib import Path
from functools import partial
import argparse
import ast
import hashlib
import json
import types
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from torchaudio.compliance import kaldi

parser=argparse.ArgumentParser(description='Full pyannote embedding backbone references with pinned WeSpeaker filterbanks.')
parser.add_argument('--models',type=Path,required=True)
parser.add_argument('--speech',type=Path,required=True)
parser.add_argument('--reference-source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);torch.set_num_interop_threads(1)
assert (np.__version__,ort.__version__,torch.__version__,torchaudio.__version__)==('2.2.4','1.29.0','2.11.0+cpu','2.11.0+cpu')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def source_sha(path):return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
asset_path=Path(__file__).with_name('assets.json')
assets=json.loads(asset_path.read_text(encoding='utf-8'))
for name,entry in assets['files'].items():
    path=args.models/name
    assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
source=args.reference_source/assets['reference']['frontend']
assert source_sha(source)==assets['reference']['frontend_lf_sha256']
tree=ast.parse(source.read_text(encoding='utf-8'))
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compute_fbank')
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
namespace=dict(torch=torch)
exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
settings=dict(num_mel_bins=80,frame_length=25.0,frame_shift=10.0,round_to_power_of_two=True,snip_edges=True,dither=0.0,sample_frequency=16000,window_type='hamming',use_energy=False)
shim=types.SimpleNamespace(hparams=types.SimpleNamespace(fbank_centering_span=None),_fbank=partial(kaldi.fbank,**settings))
opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.add_session_config_entry('session.intra_op.allow_spinning','0');opts.add_session_config_entry('session.inter_op.allow_spinning','0')
session=ort.InferenceSession(str(args.models/'embedding_encoder.onnx'),opts,providers=['CPUExecutionProvider'])
files={};cases=[]
def save(name,value):
    value=np.ascontiguousarray(value,dtype=np.float32);assert np.isfinite(value).all()
    path=out/(name+'.npy');np.save(path,value,allow_pickle=False)
    files[path.name]=dict(sha256=sha(path),bytes=path.stat().st_size,shape=list(value.shape),dtype=str(value.dtype))
    return path.name
inputs=[]
for batch,frames in [(1,200),(1,201),(1,400),(1,800),(2,200)]:
    values=(np.sin(np.arange(batch*frames*80,dtype=np.float64)*0.01)*2).astype(np.float32).reshape(batch,frames,80)
    inputs.append((f'synthetic-b{batch}-f{frames}',values,dict(kind='deterministic sin(i*.01)*2, cast to float32')))
speech_path=args.speech
speech=json.loads(speech_path.read_text(encoding='utf-8'))
for name in ('english-16k','french-44k-stereo','jfk-48k-stereo'):
    case=next(c for c in speech['cases'] if c['name']==name)
    path=speech_path.parent/case['pcm'];assert sha(path)==case['pcm_sha256']
    waveform=torch.from_numpy(np.load(path,allow_pickle=False).reshape(1,1,-1))
    features=namespace['compute_fbank'](shim,waveform).numpy()
    inputs.append((name,features,dict(kind='pinned upstream compute_fbank',pcm_sha256=sha(path),speech_manifest_sha256=sha(speech_path))))
for name,values,provenance in inputs:
    before=values.tobytes();result=session.run(None,{'fbank_features':values})[0]
    assert values.tobytes()==before and result.shape==(values.shape[0],2560,(values.shape[1]+7)//8)
    repeat=session.run(None,{'fbank_features':values})[0];assert np.array_equal(result,repeat)
    cases.append(dict(name=name,input=save(name+'-input',values),output=save(name+'-output',result),provenance=provenance))
    print(name,list(values.shape),'->',list(result.shape),flush=True)
rejections=[]
for name,feeds in [('missing',{}),('rank',{'fbank_features':np.zeros((200,80),np.float32)}),('width',{'fbank_features':np.zeros((1,200,79),np.float32)})]:
    try:session.run(None,feeds)
    except Exception as ex:rejections.append(dict(name=name,error=str(ex)))
    else:raise AssertionError('Native accepted '+name)
assert np.array_equal(session.run(None,{'fbank_features':inputs[0][1]})[0],np.load(out/cases[0]['output'],allow_pickle=False))
manifest=dict(scope='pyannote embedding backbone, not full embedding or diarization',assets=assets,cases=cases,files=files,
    numpy=np.__version__,onnxruntime=ort.__version__,torch=torch.__version__,torchaudio=torchaudio.__version__,
    generator_lf_sha256=source_sha(Path(__file__)),assets_lf_sha256=source_sha(asset_path),frontend_source_sha256=sha(source),frontend_revision=assets['reference']['revision'],
    frontend_settings=settings,native_settings=dict(provider='CPUExecutionProvider',threads=1,execution='sequential',optimization='all',spinning=False),
    rejections=rejections,scaled_absolute_tolerance=1e-4)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
