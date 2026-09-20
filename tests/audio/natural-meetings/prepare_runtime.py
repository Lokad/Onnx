"""Bind immutable products, borrowed inputs/models and qualified native sources."""
from pathlib import Path
import argparse
import importlib.metadata
import inspect
import shutil
import sys
from common import CORE,DATA,limits,pin,read,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];base=a.artifact.resolve()
    assert not (base/'manifest.json').exists() and not (base/'reference-source').exists()
    assert pin(base/'bin/Lokad.Onnx.dll')['sha256']==CORE and pin(base/'bin/Lokad.Onnx.Data.dll')['sha256']==DATA
    labels=read(base/'labels.json');audit=read(base/'input-audit.json');assert audit['passed'] is True and audit['labels']==pin(base/'labels.json')
    versions=dict(numpy='2.2.4',onnxruntime='1.29.0',torch='2.11.0+cpu',transformers='5.16.1',tokenizers='0.23.2')
    for name,version in versions.items():assert importlib.metadata.version(name)==version,name
    def copy(source,destination,sha):
        assert pin(source)['sha256']==sha,str(source)
        assert not destination.exists();destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination);assert pin(destination)==pin(source)
    old=root/'artifacts/parakeet-recording-20260919';previous=read(old/'frozen.json');receipt=read(old/'receipt.json')
    assert receipt['closed'] is True and pin(old/'frozen.json')==receipt['files']['frozen.json']
    for name in ['tests/parakeet/recording/native.py','tests/parakeet/transcribe/generate_reference.py',
                 'tests/parakeet/transcribe/assets.json','external/onnx-asr/src/onnx_asr/asr.py']:
        copy(old/'source'/name,base/'reference-source'/name,previous['source'][name])
    old=root/'artifacts/whisper-recording-v2-20260919';previous=read(old/'reference-frozen.json');receipt=read(old/'receipt.json')
    assert receipt['all_owned_workers_terminal'] is True and pin(old/'reference-frozen.json')['sha256']==receipt['files']['reference-frozen.json']
    for name in ['tests/whisper/recording/generate_reference.py','tests/whisper/recording/generate_rules.py',
                 'tests/whisper/recording/sources.json','tests/whisper/transcription-assets.json']:
        copy(old/'reference-source'/name,base/'reference-source'/name,previous['files'][name])
    inputs=read(old/'inputs/inputs.json')
    for name,item in inputs['sources'].items():copy(old/'inputs/upstream'/Path(name).name,base/'upstream/whisper'/Path(name).name,item['sha256'])
    def file(path):
        path=Path(path).resolve();return dict(path=path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix(),**pin(path))
    families={}
    for family,assets in [('parakeet',base/'reference-source/tests/parakeet/transcribe/assets.json'),
                          ('whisper',base/'reference-source/tests/whisper/transcription-assets.json')]:
        folder='models/'+('parakeet-tdt-0.6b-v3' if family=='parakeet' else 'whisper-large-v3-turbo')
        pins=read(assets);models={}
        for name,wanted in pins['files'].items():
            item=file(root/folder/name);assert {k:item[k] for k in ['bytes','sha256']}=={k:wanted[k] for k in ['bytes','sha256']},name
            models[name]=item
        families[family]=dict(model_directory=folder,models=models,assets=pins)
    cases=[]
    for c in labels['cases']+[labels['recovery']]:
        audio=file(c['audio']);assert {k:audio[k] for k in ['bytes','sha256']}==c['wav']
        cases.append(dict(name=c['name'],samples=c['samples'],audio=audio,pcm_sha256=c['pcm_sha256'],language='en',
            max_windows=256,max_tokens=4096,max_tokens_per_frame=10,max_new_tokens=444))
    import numpy,torch,onnxruntime,transformers.audio_utils,tokenizers
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
    binaries={}
    for directory in [Path(onnxruntime.__file__).parent/'capi',Path(torch.__file__).parent/'lib',Path(numpy.__file__).parent/'_core',
                      Path(numpy.__file__).parent/'fft',Path(numpy.__file__).parent.parent/'numpy.libs',Path(tokenizers.__file__).parent]:
        for path in directory.glob('*'):
            if path.is_file() and path.suffix.lower() in ['.dll','.pyd']:binaries[path.as_posix()]=pin(path)
    binaries[Path(sys.executable).as_posix()]=pin(sys.executable)
    for module in [transformers.audio_utils,WhisperFeatureExtractor]:
        path=Path(inspect.getfile(module));binaries[path.as_posix()]=pin(path)
    value=dict(schema=1,protocol='natural-meeting-recording-v1',cases=cases,families=families,versions=versions,
        core_sha256=CORE,data_sha256=DATA,labels=pin(base/'labels.json'),input_audit=pin(base/'input-audit.json'),
        native_files=binaries,limits=dict(native=limits('windows'),managed=limits('amd')),
        hosts=dict(native='Windows i7-14700KF CPU2',managed='AMD EPYC 9V74 CPU2'),
        schedule=['parakeet','whisper'],comparison=dict(public_decisions='exact, including window and segment timestamps',
            whisper_confidence='finite diagnostic values with valid probability/skip policy, as in existing recording qualification',
            intermediate_numerics='Existing scaled1e-4 tensor gates remain separate and unchanged'),
        timing_scope='One accuracy replay per engine/model; no cross-host speed comparison. Native reference includes existing validation and Parakeet upstream trajectory crosschecks.')
    write(base/'manifest.json',value);print('Bound both recording families, three borrowed PCM cases and',len(binaries),'native file identities.')


if __name__=='__main__':main()
