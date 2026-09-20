"""Verify the isolated Linux reference dependencies without neural inference."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import sys
import time


CANONICAL_BODY='43272a768eadeac0fd6eba53bee1c79ec6cf5f86aab3db8af8b6e7cab34893d2'
LINUX_DUMP_BODY='e8548ecf6b3c4ee12d748d37e458eafc18628f6e040b4f3c54dd145bacb60f99'


def canonical(node):
    if isinstance(node,ast.AST):
        return dict(type=type(node).__name__,fields={name:canonical(value) for name,value in ast.iter_fields(node)})
    if isinstance(node,list):return [canonical(n) for n in node]
    return node


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','artifact','original']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();base=a.artifact.resolve();original=a.original.resolve()
    assert sys.platform=='linux' and sys.version_info[:2]==(3,12)
    assert not (base/'dependency-check.json').exists()
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    os.sched_setaffinity(0,{2})
    sys.path[:0]=[str(base/'python'),str(root/'artifacts/asr-multilingual-amd-20260920/python'),
        str(original/'runtime'),str(original/'reference-source/tests/whisper/recording')]
    from common import pin,load,read,write
    frozen=read(original/'frozen.json')
    for name,wanted in frozen['files'].items():assert pin(original/name)==wanted,name
    import numpy as np
    import torch
    import onnxruntime
    import transformers
    import tokenizers
    import psutil
    import sympy
    import networkx
    import mpmath
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
    import transformers.audio_utils
    import whisper_recording
    versions={n:importlib.metadata.version(n) for n in ['numpy','torch','onnxruntime','transformers','tokenizers','sympy',
        'networkx','mpmath','filelock','typing_extensions','jinja2','fsspec','setuptools']}
    expected=read(original/'manifest.json')['versions'];assert all(versions[n]==v for n,v in expected.items())
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.version.cuda is None and torch.get_num_threads()==torch.get_num_interop_threads()==1
    x=torch.tensor([1.,2.,3.],dtype=torch.float32)
    assert x.device.type=='cpu' and torch.isfinite(torch.logsumexp(x,dim=0)).item()
    generator=original/'reference-source/tests/whisper/recording/generate_reference.py'
    helper=load('linux_original_whisper_generator',generator)
    _,_,body=whisper_recording.functions(generator,dict(vars(helper)))
    assert body==LINUX_DUMP_BODY
    tree=ast.parse(generator.read_text(encoding='utf-8'))
    main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes=[n for n in main_node.body if isinstance(n,ast.FunctionDef) and n.name in ['decode_window','segments_for']]
    canonical_sha=hashlib.sha256(json.dumps(canonical(ast.Module(body=nodes,type_ignores=[])),sort_keys=True).encode()).hexdigest()
    assert canonical_sha==CANONICAL_BODY
    helper.upstream(original/'upstream/whisper/decoding.py')
    helper.seek_program(original/'upstream/whisper/transcribe.py')
    identities={}
    for directory in [Path(onnxruntime.__file__).parent/'capi',Path(torch.__file__).parent/'lib',Path(torch.__file__).parent,
        Path(np.__file__).parent/'_core',Path(np.__file__).parent/'fft',Path(np.__file__).parent.parent/'numpy.libs',Path(tokenizers.__file__).parent]:
        for path in directory.glob('*'):
            if path.is_file() and '.so' in path.name:identities[str(path.resolve())]=pin(path)
    identities[str(Path(sys.executable).resolve())]=pin(Path(sys.executable).resolve())
    windows=read(original/'manifest.json')['native_files'];frontend={}
    for module in [transformers.audio_utils,WhisperFeatureExtractor]:
        path=Path(inspect.getfile(module));item=pin(path)
        expected=[v for p,v in windows.items() if Path(p).name==path.name]
        assert len(expected)==1 and item==expected[0],path.name
        identities[str(path.resolve())]=item;frontend[path.name]=item
    value=dict(passed=True,created=time.time(),python=sys.version,platform=platform.platform(),libc=platform.libc_ver(),versions=versions,
        expected_model_versions=read(original/'manifest.json')['versions'],torch_cpu_only=True,threads=1,affinity=sorted(os.sched_getaffinity(0)),
        available=psutil.virtual_memory().available,free=shutil.disk_usage(base).free,native_files=identities,unchanged_frontend_sources=frontend,
        borrowed_function_body_sha256=body,portable_function_body_sha256=canonical_sha,generator=pin(generator),checker=pin(Path(__file__)),
        scope='Dependency import, CPU/thread/affinity and unchanged original function-body checks only; no neural inference')
    write(base/'dependency-check.json',value)
    print(json.dumps(value))


if __name__=='__main__':main()
