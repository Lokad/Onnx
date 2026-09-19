from pathlib import Path
from functools import partial
import argparse,ast,hashlib,inspect,json,types
import numpy as np
import torch
import torchaudio
from torchaudio.compliance import kaldi

parser=argparse.ArgumentParser(description='Complete pinned native WeSpeaker frontend references; no model downloads.')
parser.add_argument('--reference-source',type=Path,required=True)
parser.add_argument('--speech',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=False)
pins_path=Path(__file__).with_name('pins.json');pins=json.loads(pins_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=args.reference_source/'src/pyannote/audio/models/embedding/wespeaker/__init__.py'
assert hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='a2c13a792c50d97f7a7583b69fcabd5eb5efc34d1374c2b6e5757bbcf3b62139'
assert (np.__version__,torch.__version__,torchaudio.__version__)==('2.2.4','2.11.0+cpu','2.11.0+cpu')
torch.set_num_threads(1);torch.set_num_interop_threads(1)
method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='compute_fbank')
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
namespace={'torch':torch};exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
settings=dict(num_mel_bins=80,frame_length=25.0,frame_shift=10.0,round_to_power_of_two=True,snip_edges=True,dither=0.0,sample_frequency=16000,window_type='hamming',use_energy=False)
shim=types.SimpleNamespace(hparams=types.SimpleNamespace(fbank_centering_span=None),_fbank=partial(kaldi.fbank,**settings))
files={};cases=[]
def save(name,array):
    a=np.ascontiguousarray(array,np.float32);assert np.isfinite(a).all(),name
    p=out/(name+'.npy');np.save(p,a,allow_pickle=False)
    files[p.name]=dict(sha256=sha(p),shape=list(a.shape),dtype=str(a.dtype),bytes=p.stat().st_size);return p.name
window=kaldi._feature_window_function('hamming',400,.42,torch.device('cpu'),torch.float32)
mel,_=kaldi.get_mel_banks(80,512,16000,20,0,100,-500,1)
save('window',window.numpy());save('mel',mel.numpy())
def noise(length):
    state=20260919;values=np.empty(length,np.float32)
    for i in range(length):
        state=(1664525*state+1013904223)&0xffffffff
        values[i]=(((state>>8)&65535)-32768)/131072
    return values
inputs=[]
for length in (400,559,560,561,16000,160000,480000):inputs.append((f'noise-{length}',noise(length),'deterministic LCG'))
for length in (400,560,16000,160000):
    inputs.append((f'silence-{length}',np.zeros(length,np.float32),'digital silence'))
    inputs.append((f'dc-{length}',np.full(length,.25,np.float32),'constant .25'))
inputs.append(('quiet',noise(16000)*np.float32(1e-6),'quiet LCG'))
impulse=np.zeros(16000,np.float32);impulse[[0,199,400,8000,15999]]=[.5,-.25,.75,-.5,.25]
inputs.append(('impulse',impulse,'five impulses'))
t=np.arange(16000,dtype=np.float64)/16000
inputs.append(('tones',(.2*np.sin(2*np.pi*440*t)+.1*np.sin(2*np.pi*1000*t)+.05*np.sin(2*np.pi*3100*t)).astype(np.float32),'three tones'))
speech_path=args.speech;speech=json.loads(speech_path.read_text())
for name in ('english-16k','french-44k-stereo','jfk-48k-stereo'):
    case=next(c for c in speech['cases'] if c['name']==name);p=speech_path.parent/case['pcm'];assert sha(p)==case['pcm_sha256']
    inputs.append((name,np.load(p).reshape(-1),dict(pcm_sha256=sha(p),speech_manifest_sha256=sha(speech_path))))
for name,samples,provenance in inputs:
    waveform=torch.from_numpy(samples.reshape(1,1,-1));before=waveform.clone()
    features=namespace['compute_fbank'](shim,waveform)
    assert torch.equal(waveform,before) and torch.equal(features,namespace['compute_fbank'](shim,waveform))
    scaled=waveform[0]*32768;raw=kaldi.fbank(scaled,**settings)
    framed,_=kaldi._get_window(scaled[0],512,400,160,'hamming',.42,True,True,1.,0.,True,.97)
    powers=torch.fft.rfft(framed[:3]).abs().pow(2)
    expected=1+(len(samples)-400)//160;assert tuple(features.shape)==(1,expected,80)
    cases.append(dict(name=name,provenance=provenance,input=save(name+'-pcm',samples),output=save(name+'-features',features.numpy()),
        raw=save(name+'-raw',raw.numpy()),windowed=save(name+'-windowed',framed[:3].numpy()),power=save(name+'-power',powers.numpy())))
    print(name,list(features.shape),flush=True)
rejections=[]
for length in (0,1,399):
    try:namespace['compute_fbank'](shim,torch.zeros((1,1,length)))
    except Exception as ex:rejections.append(dict(length=length,error=str(ex)))
    else:raise AssertionError('Native accepted undersized audio')
kaldi_path=Path(inspect.getfile(kaldi))
assert hashlib.sha256(kaldi_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pins['kaldi_lf_sha256']
assert [c['name'] for c in cases]==pins['cases'] and settings==pins['settings']
record=dict(scope='Full pinned pyannote WeSpeaker filterbank outputs',numpy=np.__version__,torch=torch.__version__,torchaudio=torchaudio.__version__,
    generator_lf_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),pins=pins,pins_lf_sha256=hashlib.sha256(pins_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),frontend_source_sha256=sha(source),kaldi_source_sha256=sha(kaldi_path),settings=settings,cases=cases,files=files,rejections=rejections,tolerance=1e-4)
(out/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(out/'kaldi.py.txt').write_bytes(kaldi_path.read_bytes());print('Complete',len(cases),'cases')
