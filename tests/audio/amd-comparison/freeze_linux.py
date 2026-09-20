"""Resolve actual Linux native sources/libraries before any benchmark inference."""
from pathlib import Path
import hashlib, importlib, importlib.metadata, inspect, os, sys
from protocol import pin,read,write,LIMITS,FAMILIES


def main():
    assert os.name=='posix' and os.sched_getaffinity(0)=={0}
    base=Path(sys.argv[1]).resolve();assert not (base/'frozen.json').exists()
    prepared=read(base/'preparation.json')
    for name,wanted in prepared['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in prepared['remote_models'].items():assert pin(Path(name))==wanted,name
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:assert os.environ[key]=='1'
    assert not any(k.lower().startswith(('lokad_','dotnet_','complus_')) for k in os.environ)
    python_paths=os.environ['PYTHONPATH'].split(os.pathsep)
    assert len(python_paths)==3 and all(Path(p).is_dir() for p in python_paths)
    versions=dict(numpy='2.2.4',onnxruntime='1.29.0',torch='2.11.0+cpu',torchaudio='2.11.0+cpu',scipy='1.16.3',
                  einops='0.8.1',**{'pyannote.core':'6.0.1'},sortedcontainers='2.4.0',pandas='2.2.3',transformers='5.16.1',tokenizers='0.23.2',psutil='7.0.0')
    for name,version in versions.items():assert importlib.metadata.version(name)==version,name
    modules={name:importlib.import_module(name) for name in ['numpy','onnxruntime','torch','torchaudio','scipy','einops','pyannote.core','sortedcontainers','pandas','transformers','tokenizers','psutil']}
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.get_num_threads()==torch.get_num_interop_threads()==1
    from torchaudio.compliance import kaldi
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
    import transformers.audio_utils
    import onnxruntime.capi.onnxruntime_pybind11_state as ort_binary
    capi=Path(ort_binary.__file__).parent
    ort_files={p.name:dict(path=str(p),**pin(p)) for p in sorted(capi.iterdir()) if p.is_file() and '.so' in p.name}
    assert ort_files
    # Inventory complete installed package roots and their sibling numerical libraries.
    roots=set()
    for name,module in modules.items():
        if name=='pyannote.core':roots.add(Path(module.__file__).parent.parent)
        else:roots.add(Path(module.__file__).parent)
    for folder in list(roots):
        for extra in folder.parent.glob(folder.name+'.libs'):
            if extra.is_dir():roots.add(extra)
    for name in versions:
        metadata=importlib.metadata.distribution(name)
        for file in metadata.files or []:
            if str(file).endswith(('.dist-info/METADATA','.dist-info/RECORD')):roots.add(Path(metadata.locate_file(file)).parent)
    external=dict(prepared['remote_models'])
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.pyc','.pyo']:external[str(path)]=pin(path)
    external[str(Path(sys.executable))]=pin(Path(sys.executable))
    (base/'manifests').mkdir()
    for family in FAMILIES:
        manifest=read(base/'draft-manifests'/(family+'.json'));manifest['native_binaries']=ort_files
        if family=='pyannote':assert hashlib.sha256(Path(inspect.getfile(kaldi)).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==manifest['pins']['kaldi_lf_sha256']
        if family=='whisper':
            for name,obj in [('extractor',WhisperFeatureExtractor),('audio_utils',transformers.audio_utils)]:
                path=Path(inspect.getfile(obj));assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==manifest['upstream_expected_lf'][name]
                manifest['native_sources'][name]=dict(path=str(path),**pin(path));external[str(path)]=pin(path)
        for item in [*manifest['models'].values(),*manifest['native_sources'].values(),manifest['reference']]:
            path=base/'assets'/item['path'];assert pin(path)=={k:item[k] for k in ['bytes','sha256']},str(path)
        write(base/'manifests'/(family+'.json'),manifest)
    write(base/'native-environment.json',dict(versions=versions,python=sys.version,interpreter=pin(Path(sys.executable)),python_paths=python_paths,
                                           torch_parallel=torch.__config__.parallel_info(),torch_config=torch.__config__.show(),roots=sorted(map(str,roots))))
    frozen=dict(schema=1,source=prepared['source'],product_source='1d10d22',limits=LIMITS,python_paths=python_paths,interpreter=pin(Path(sys.executable)),
                external=external,files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file() and p.name not in ['payload.tar.gz']})
    write(base/'frozen.json',frozen)
    print(__import__('json').dumps(dict(frozen=pin(base/'frozen.json'),files=len(frozen['files']),external=len(external),versions=versions)))


if __name__=='__main__':main()
