"""Independent complete PCM applications; no managed output enters native inference."""
from pathlib import Path
import argparse
import ctypes
import importlib.metadata
import importlib.util
import json
import os
import time
from common import read, sha, pin, write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','manifest','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();root=args.root.resolve();manifest=read(args.manifest)
    assert os.name=='nt' and not args.output.exists()
    mask,system_mask=ctypes.c_size_t(),ctypes.c_size_t()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    assert kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),ctypes.byref(mask),ctypes.byref(system_mask)) and mask.value==4
    settings={k:v for k,v in os.environ.items() if k.lower().startswith(('lokad_','dotnet_','complus_'))};assert not settings
    for name,version in manifest['versions'].items():assert importlib.metadata.version(name)==version,name
    def verify(spec):
        path=root/spec['path'];assert pin(path)=={k:spec[k] for k in ('bytes','sha256')},path;return path
    for spec in manifest['models'].values():verify(spec)
    for spec in manifest['source_files'].values():verify(spec)
    for spec in manifest['native_binaries'].values():verify(spec)
    audio_path=verify(manifest['audio']);audio=read(audio_path)
    assert audio['protocol']=='multilingual-noise-asr-v2' and len(audio['cases'])==40
    import numpy as np
    import onnxruntime as ort
    family=manifest['family'];assert family in ('parakeet','whisper')
    if family=='parakeet':
        path=root/manifest['source_files']['parakeet_adapter']['path']
        spec=importlib.util.spec_from_file_location('pinned_parakeet_adapter',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);adapter=module.Parakeet
    else:
        from whisper_adapter import Whisper
        adapter=Whisper
    args.output.mkdir(parents=True);start=time.perf_counter();model=adapter(root,manifest);constructor=time.perf_counter()-start
    rows=[];held=[]
    for index,case in enumerate(audio['cases']+audio['cases'][:1]):
        pcm_path=audio_path.parent/case['pcm'];assert sha(pcm_path)==case['pcm_sha256']
        pcm=np.load(pcm_path,allow_pickle=False);assert pcm.dtype==np.float32 and pcm.shape==(case['samples'],) and np.isfinite(pcm).all() and np.abs(pcm).max()<=1
        original=pcm.tobytes();start=time.perf_counter_ns()
        decision=model(pcm) if family=='parakeet' else model(pcm,case['language'])
        end=time.perf_counter_ns();held.append((pcm,original,decision,json.dumps(decision,sort_keys=True)))
        for values,bits,result,saved in held:assert values.tobytes()==bits and json.dumps(result,sort_keys=True)==saved
        row=dict(name=case['name'],language=case['language'],repeat=index==40,decision=decision,
                 seconds=(end-start)/1e9,start_ticks=start,end_ticks=end,frequency=1000000000,
                 pcm_sha256=sha(pcm_path),input_and_held_results_unchanged=True)
        rows.append(row);write_new(args.output/f'{index:02d}.json',row)
        print(index,case['name'],case['language'],row['seconds'],flush=True)
    import onnxruntime.capi.onnxruntime_pybind11_state as ort_binary
    write_new(args.output/'result.json',dict(schema=1,protocol='multilingual-noise-asr-v2',family=family,engine='ort',passed=True,
        manifest_sha256=sha(args.manifest),audio_sha256=sha(audio_path),affinity=mask.value,flags=settings,
        numpy=np.__version__,onnxruntime=ort.__version__,ort_binary=pin(Path(ort_binary.__file__)),
        constructor_seconds=constructor,cases=rows))


if __name__=='__main__':main()
