"""Prepare exact audio workloads and portable consumers without using the VM."""
from pathlib import Path
import copy, hashlib, json, shutil, subprocess, tarfile
import numpy as np
from protocol import pin,read,write,FAMILIES,LIMITS

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/audio-amd-comparison-20260920'
REMOTE='/home/vermorel/Onnx/artifacts/audio-amd-comparison-20260920'


def main():
    assert not BASE.exists();BASE.mkdir();stage=BASE/'payload';stage.mkdir()
    for folder in ['bin','runtime','assets','draft-manifests']:(stage/folder).mkdir()
    product=ROOT/'artifacts/wespeaker-frame-product-amd-v4-20260920';qualified=read(product/'closed.json')
    assert pin(product/'closed.json')['sha256']=='d02b755dea95632828e1e886fa6ed8b80039cfd4a6454d46de50fcb74297c75a'
    bindings={};old_bases={family:ROOT/'artifacts'/('whisper-ort-baseline-20260919' if family=='whisper' else 'audio-ort-baseline-v2-20260919') for family in FAMILIES}
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll','FastBertTokenizer.dll','Lokad.Tokenizers.dll','SixLabors.ImageSharp.dll']:
        source=product/'collected/result/product-bin'/name;wanted=qualified['files'][source.relative_to(ROOT).as_posix()]
        assert pin(source)==wanted;shutil.copyfile(source,stage/'bin'/name);bindings[source.relative_to(ROOT).as_posix()]=wanted
    for family,runner in [('parakeet','AudioBenchmark'),('whisper','WhisperBenchmark')]:
        base=old_bases[family];frozen=read(base/'conformance/frozen.json')
        for suffix in ['.dll','.deps.json','.runtimeconfig.json']:
            source=base/'bin'/(runner+suffix);relative=source.relative_to(ROOT).as_posix()
            assert pin(source)==frozen['files'][relative];shutil.copyfile(source,stage/'bin'/source.name);bindings[relative]=pin(source)
    def asset(source,name,wanted=None):
        source=Path(source);actual=pin(source)
        if wanted is not None:assert actual=={k:wanted[k] for k in ['bytes','sha256']},str(source)
        target=stage/'assets'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert pin(target)==actual
        else:shutil.copyfile(source,target)
        bindings[source.relative_to(ROOT).as_posix()]=actual
        return dict(path=name,**actual)
    natural=read(ROOT/'artifacts/pyannote-natural-meetings-20260920/manifest.json')
    model_bindings={};counts={};sizes={}
    for family in FAMILIES:
        original=old_bases[family]/'inputs'/(family+'.json');manifest=read(original)
        frozen=read(old_bases[family]/'conformance/frozen.json');assert pin(original)==frozen['files'][original.relative_to(ROOT).as_posix()]
        bindings[original.relative_to(ROOT).as_posix()]=pin(original)
        manifest['reference']=asset(ROOT/manifest['reference']['path'],'reference/'+family+'.json',manifest['reference'])
        for name,item in manifest['models'].items():
            actual=pin(ROOT/item['path']);assert actual=={k:item[k] for k in ['bytes','sha256']}
            bindings[item['path']]=actual
            remote_path=natural['models'][name]['amd_path'] if family=='pyannote' else item['path']
            item['path']='/home/vermorel/Onnx/'+remote_path;model_bindings[item['path']]=actual
        for case in manifest['cases']:
            source=ROOT/case['pcm']['path'];pcm=np.load(source,allow_pickle=False)
            assert pcm.dtype==np.float32 and pcm.shape==(case['samples'],) and np.isfinite(pcm).all()
            case['pcm']=asset(source,'pcm/'+case['pcm']['sha256']+'.npy',case['pcm']);case['raw_sha256']=hashlib.sha256(pcm.tobytes()).hexdigest()
            if family=='whisper':case['features']=asset(ROOT/case['features']['path'],'features/'+case['features']['sha256']+'.npy',case['features'])
        manifest['native_sources']={}
        if family=='parakeet':
            manifest['upstream']=asset(ROOT/manifest['upstream']['path'],'upstream/asr.py',manifest['upstream'])
            manifest['native_sources']['asr']=manifest['upstream']
            manifest['native_versions']=dict(numpy='2.2.4',onnxruntime='1.29.0')
        elif family=='pyannote':
            for name,item in list(manifest['upstream'].items()):manifest['upstream'][name]=asset(ROOT/item['path'],'upstream/pyannote/'+name,item)
            manifest['native_sources'].update(manifest['upstream'])
            for name,item in list(manifest['native_assets'].items()):manifest['native_assets'][name]=asset(ROOT/item['path'],'native-assets/'+name,item)
            rules=ROOT/'tests/pyannote/diarization/native_rules.py'
            manifest['native_sources']['native_rules']=asset(rules,'tests/pyannote/diarization/native_rules.py',frozen['files'][rules.relative_to(ROOT).as_posix()])
            manifest['native_sources'].update(manifest['native_assets'])
            manifest['native_versions']=dict(manifest['pins']['versions'],pandas='2.2.3')
        else:
            # Platform-specific extractor/audio_utils paths are resolved and pinned on Linux before freezing.
            manifest['upstream_expected_lf']={name:hashlib.sha256((ROOT/item['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name,item in manifest['upstream'].items() if name in ['extractor','audio_utils']}
            generator=manifest['upstream']['reference_generator']
            manifest['native_sources']['reference_generator']=asset(ROOT/generator['path'],'reference/whisper-generator.py',generator)
            manifest['audio_manifest']=asset(ROOT/manifest['audio_manifest']['path'],'reference/audio.json',manifest['audio_manifest'])
            manifest['native_sources']['audio_manifest']=manifest['audio_manifest']
            manifest['native_versions']=dict(manifest['versions'])
            manifest.pop('upstream')
        adapter=ROOT/('tests/audio/whisper-comparison/native_adapters.py' if family=='whisper' else 'tests/audio/comparison/native_adapters.py')
        adapter_name='whisper_adapter.py' if family=='whisper' else 'audio_adapter.py'
        wanted=frozen['files'][adapter.relative_to(ROOT).as_posix()];assert pin(adapter)==wanted
        target=stage/'runtime'/adapter_name
        if not target.exists():shutil.copyfile(adapter,target)
        else:assert pin(target)==wanted
        manifest['adapter']=dict(path='../runtime/'+adapter_name,**wanted);manifest['native_sources']['adapter']=manifest['adapter']
        bindings[adapter.relative_to(ROOT).as_posix()]=wanted
        manifest.update(native_binaries={},warmup_passes=1,measured_passes=3,product_source='1d10d22f73282bb00630371887ca3719d2e1553b',
                        core_sha256=pin(stage/'bin/Lokad.Onnx.dll')['sha256'],data_sha256=pin(stage/'bin/Lokad.Onnx.Data.dll')['sha256'])
        write(stage/'draft-manifests'/(family+'.json'),manifest);counts[family]=len(manifest['cases']);sizes[family]=sum(c['samples'] for c in manifest['cases'])
    assert counts==dict(parakeet=20,pyannote=4,whisper=20) and sizes['parakeet']==sizes['whisper']==3412240
    for name in ['native.py','protocol.py']:shutil.copyfile(Path(__file__).with_name(name),stage/'runtime'/name)
    shutil.copyfile(ROOT/'eng/campaign_processes.py',stage/'runtime/campaign_processes.py')
    shutil.copyfile(ROOT/'.agent/m5-audio-amd-baselines-20260920.md',stage/'prospective-plan.md')
    write(stage/'preparation.json',dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),product_source='1d10d22',
        bindings=bindings,remote_models=model_bindings,counts=counts,samples=sizes,limits=LIMITS,
        files={p.relative_to(stage).as_posix():pin(p) for p in sorted(stage.rglob('*')) if p.is_file()}))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for path in sorted(stage.rglob('*')):
            if path.is_file():tar.add(path,arcname=path.relative_to(stage).as_posix(),recursive=False)
    write(BASE/'transfer.json',dict(archive=pin(BASE/'payload.tar.gz'),preparation=pin(stage/'preparation.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),counts=counts,samples=sizes,bindings=len(bindings))))


if __name__=='__main__':main()
