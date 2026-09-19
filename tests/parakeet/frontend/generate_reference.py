"""Pinned native Parakeet frontend: complete features from PCM, with no ASR claim."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import onnx
import onnxruntime as ort

MODEL_SHA = 'a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--speech-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert (np.__version__, onnx.__version__, ort.__version__) == ('2.2.4','1.22.0','1.29.0')
    assert args.model.stat().st_size == 139764 and sha(args.model) == MODEL_SHA
    args.output.mkdir(parents=True, exist_ok=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 4
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    session = ort.InferenceSession(str(args.model), options, providers=['CPUExecutionProvider'])
    files, cases = {}, []
    def save(name, value):
        path = args.output / (name + '.npy')
        np.save(path, value, allow_pickle=False)
        files[path.name] = dict(dtype=str(value.dtype), shape=list(value.shape), bytes=path.stat().st_size, sha256=sha(path))
        return path.name
    def run(name, pcm, lengths, failure=False):
        inputs = dict(waveforms=np.ascontiguousarray(pcm, dtype=np.float32), waveforms_lens=np.array(lengths,dtype=np.int64))
        bound = {k:save(name+'-in-'+k,v) for k,v in inputs.items()}
        try: outputs=session.run(None,inputs)
        except ort.capi.onnxruntime_pybind11_state.InvalidArgument as exc:
            assert failure
            cases.append(dict(name=name,inputs=bound,expected_failure='invalid-input',native_error=str(exc)))
            return
        assert not failure
        assert all(np.isfinite(v).all() for v in outputs), name
        expected_lengths=np.array(lengths,dtype=np.int64)//160+1
        assert np.array_equal(outputs[1],expected_lengths)
        assert outputs[0].shape == (pcm.shape[0],128,pcm.shape[1]//160+1)
        for b,length in enumerate(expected_lengths): assert np.count_nonzero(outputs[0][b,:,length:]) == 0
        cases.append(dict(name=name,inputs=bound,outputs={v.name:save(name+'-out-'+v.name,x) for v,x in zip(session.get_outputs(),outputs)}))
        print(name,outputs[0].shape,flush=True)
    rng=np.random.default_rng(20260919)
    for length in (257,511,512,16000,32001):
        run('noise-'+str(length),rng.normal(0,.2,(1,length)).astype(np.float32),[length])
    run('silence',np.zeros((1,16000),np.float32),[16000])
    time=np.arange(16000,dtype=np.float64)/16000
    run('tones',(.2*np.sin(2*np.pi*440*time)+.1*np.sin(2*np.pi*1234*time)).astype(np.float32)[None,:],[16000])
    run('impulses',np.eye(1,16000,8000,dtype=np.float32),[16000])
    pcm=rng.normal(0,.1,(2,32000)).astype(np.float32)
    pcm[0,16000:]=0; pcm[1,23001:]=0
    run('batch-masked',pcm,[16000,23001])
    speech=json.loads(args.speech_manifest.read_text(encoding='utf-8'))
    assert speech['sample_rate']==16000
    selected=('english-16k','french-44k-stereo','jfk-48k-stereo')
    speech_sources=[]
    for name in selected:
        case=next(c for c in speech['cases'] if c['name']==name)
        path=args.speech_manifest.parent/case['pcm']
        assert sha(path)==case['pcm_sha256']
        pcm=np.load(path,allow_pickle=False)
        assert pcm.dtype==np.float32 and pcm.ndim==1 and np.isfinite(pcm).all()
        run(name,pcm[None,:],[len(pcm)])
        speech_sources.append(case)
    run('invalid-rank',np.zeros(16000,np.float32),[16000],True)
    first=np.load(args.output/cases[0]['inputs']['waveforms'],allow_pickle=False)
    run('recovery',first,[257])
    assert all(np.array_equal(np.load(args.output/cases[0]['outputs'][k],allow_pickle=False),
                              np.load(args.output/cases[-1]['outputs'][k],allow_pickle=False)) for k in cases[0]['outputs'])
    native=Path(ort.__file__).parent/'capi'/'onnxruntime.dll'
    if not native.exists(): native=next((Path(ort.__file__).parent/'capi').glob('libonnxruntime.so.*'))
    result=dict(schema=1,scope='parakeet-frontend',repository='istupakov/parakeet-tdt-0.6b-v3-onnx',
        revision='8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce',model_sha256=MODEL_SHA,
        generator_lf_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        numpy=np.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__,native_sha256=sha(native),
        native_settings=dict(provider='CPUExecutionProvider',threads=1,execution='sequential',optimization='all',spinning=False),
        speech_manifest_sha256=sha(args.speech_manifest),speech_sources=speech_sources,files=files,cases=cases)
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
