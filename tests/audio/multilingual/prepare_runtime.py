"""Bind qualified binaries, existing model files and native adapter source."""
from pathlib import Path
import argparse
import inspect
import importlib.metadata
from common import CORE, DATA, pin, read, sha, write_new


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];base=a.artifact.resolve();out=base/'manifests';assert not out.exists()
    assert sha(base/'bin/Lokad.Onnx.dll')==CORE and sha(base/'bin/Lokad.Onnx.Data.dll')==DATA
    def file(path):
        path=Path(path);path=path if path.is_absolute() else root/path
        return dict(path=path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),**pin(path))
    audit=read(base/'input-audit.json');assert audit['passed'] and sha(base/'inputs/audio.json')==audit['audio_sha256']
    for name,wanted in audit['files'].items():assert pin(base/'inputs'/name)==wanted,name
    versions=dict(numpy='2.2.4',onnxruntime='1.29.0',transformers='5.16.1',tokenizers='0.23.2')
    for name,version in versions.items():assert importlib.metadata.version(name)==version,name
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
    import transformers.audio_utils
    import onnxruntime.capi.onnxruntime_pybind11_state as ort_binary
    native_binaries={p.name:file(p) for p in sorted(Path(ort_binary.__file__).parent.iterdir()) if p.suffix in ('.dll','.pyd','.so') or '.so.' in p.name}
    out.mkdir()
    for family,folder,assets_path in [('parakeet','models/parakeet-tdt-0.6b-v3','tests/parakeet/transcribe/assets.json'),
                                      ('whisper','models/whisper-large-v3-turbo','tests/whisper/transcription-assets.json')]:
        staged_assets=Path(__file__).with_name(family+'-assets.json')
        assets=read(staged_assets if staged_assets.exists() else root/assets_path);models={}
        for name,wanted in assets['files'].items():
            spec=file(root/folder/name);assert {k:spec[k] for k in ('bytes','sha256')}=={k:wanted[k] for k in ('bytes','sha256')},name
            models[name]=spec
        parakeet_adapter=Path(__file__).with_name('native_adapters.py')
        if not parakeet_adapter.exists():parakeet_adapter=root/'tests/audio/comparison/native_adapters.py'
        sources=dict(extractor=file(inspect.getfile(WhisperFeatureExtractor)),audio_utils=file(inspect.getfile(transformers.audio_utils)),
            parakeet_adapter=file(parakeet_adapter),whisper_adapter=file(Path(__file__).with_name('whisper_adapter.py')))
        value=dict(schema=1,protocol='multilingual-noise-asr-v2',family=family,model_directory=folder,models=models,assets=assets,
                   versions=versions,audio=file(base/'inputs/audio.json'),source_files=sources,native_binaries=native_binaries)
        if family=='parakeet':
            upstream=Path(__file__).with_name('asr.py')
            if not upstream.exists():upstream=root/'external/onnx-asr/src/onnx_asr/asr.py'
            value['graphs']=assets['graphs'];value['upstream']=file(upstream)
            import hashlib
            assert hashlib.sha256((root/value['upstream']['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==assets['reference']['asr_lf_sha256']
            value['source_files']['asr']=value['upstream']
        write_new(out/(family+'.json'),value)
    print('Bound both model families, qualified core/Data and pinned native adapter dependencies.')


if __name__=='__main__':main()
