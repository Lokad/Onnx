"""Complete native recording applications with independently advanced state."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import wave
from common import NAMES,load,pin,read,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','artifact','output']:p.add_argument(name,type=Path)
    p.add_argument('family',choices=['parakeet','whisper']);p.add_argument('mode',choices=['inputs','run']);a=p.parse_args()
    assert sys.platform=='linux' and os.sched_getaffinity(0)=={2}, 'Linux CPU2 required before imports'
    root,base=a.root.resolve(),a.artifact.resolve();manifest=read(base/'manifest.json');family=manifest['families'][a.family]
    for name,version in manifest['versions'].items():assert importlib.metadata.version(name)==version,name
    for item in family['models'].values():assert pin(root/item['path'])=={k:item[k] for k in ['bytes','sha256']}
    for name,wanted in manifest['native_files'].items():assert pin(Path(name))==wanted,name
    import numpy as np
    import onnxruntime as ort
    cases=[]
    for c in manifest['cases']:
        path=root/c['audio']['path'];assert pin(path)=={k:c['audio'][k] for k in ['bytes','sha256']}
        with wave.open(str(path),'rb') as source:
            assert (source.getnchannels(),source.getsampwidth(),source.getframerate())==(1,2,16000)
            pcm=np.frombuffer(source.readframes(c['samples']),dtype='<i2').astype(np.float32)/np.float32(32768)
        assert pcm.shape==(c['samples'],) and hashlib.sha256(pcm.tobytes()).hexdigest()==c['pcm_sha256']
        cases.append((c,pcm,pcm.tobytes()))
    assert [c['name'] for c,_,_ in cases]==NAMES;a.output.mkdir()
    if a.mode=='inputs':
        write(a.output/'inputs.json',dict(passed=True,family=a.family,cases=[dict(name=c['name'],samples=len(pcm),pcm_sha256=c['pcm_sha256']) for c,pcm,_ in cases],affinity=4));return
    started=time.perf_counter()
    if a.family=='parakeet':
        helper=load('qualified_parakeet_recording',base/'reference-source/tests/parakeet/recording/native.py')
        model=helper.Native(root/family['model_directory'],base/'reference-source/external/onnx-asr/src/onnx_asr/asr.py')
    else:
        from whisper_recording import WhisperRecording
        model=WhisperRecording(root,base,manifest)
    setup=time.perf_counter()-started;held=[];rows=[]
    for i,(case,pcm,original) in enumerate(cases):
        assert pcm.tobytes()==original and all(json.dumps(v,sort_keys=True,allow_nan=False)==saved for v,saved in held)
        start=time.perf_counter_ns();result=model.recording(pcm,case);end=time.perf_counter_ns()
        held.append((result,json.dumps(result,sort_keys=True,allow_nan=False)));assert pcm.tobytes()==original
        row=dict(name=case['name'],seconds=(end-start)/1e9,start_ticks=start,end_ticks=end,frequency=1000000000,
            result=result,input_sha256=case['pcm_sha256'],ownership=True)
        rows.append(row);write(a.output/f'{i:02d}.json',row);print('Complete',a.family,case['name'],row['seconds'],flush=True)
    assert all(pcm.tobytes()==original for _,pcm,original in cases)
    assert all(json.dumps(v,sort_keys=True,allow_nan=False)==saved for v,saved in held)
    details=dict(original_decoder_crosschecks=model.crosschecks) if a.family=='parakeet' else dict(
        borrowed_function_body_sha256=model.body_sha256,checked_arrays=model.array_count,checked_values=model.value_count)
    write(a.output/'result.json',dict(schema=1,engine='ort',family=a.family,records=rows,setup_seconds=setup,held_outputs_unchanged=True,
        manifest_sha256=pin(base/'manifest.json')['sha256'],runner_sha256=pin(Path(__file__))['sha256'],
        python=sys.version,python_binary=pin(sys.executable),versions=manifest['versions'],affinity=4,reference_checks=details,
        flags={k:v for k,v in os.environ.items() if k.startswith(('LOKAD_','DOTNET_','COMPlus_','OMP_','MKL_','OPENBLAS_'))},
        native_settings=dict(provider='CPUExecutionProvider',intra_threads=1,inter_threads=1,sequential=True,graph_optimizations='all',spinning=False)))


if __name__=='__main__':main()
