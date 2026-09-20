"""Bind every reused baseline/input to the original closed corpus before new inference."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
from common import ROOT,CORE,pin,read,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);args=p.parse_args();base=args.artifact.resolve()
    assert not (base/'manifest.json').exists()
    prior=ROOT/'artifacts/whisper-numerical-20260919';corpus=ROOT/'artifacts/asr-labeled-20260919'
    assert pin(prior/'closed.json')['sha256']=='2f34a28b807fa97cb992924abf7372b5034385ae5cd6fd7c114512e91f610cb5'
    closed=read(prior/'closed.json');receipt=read(prior/'receipt.json');manifest=read(prior/'manifest.json');result=read(prior/'result/result.json')
    assert pin(prior/'receipt.json')['sha256']==closed['numerical_receipt_sha256']=='519cd75cb6383056d10d2be015dce92880b18a308d532f5bf5c7043db7ad7899'
    assert pin(prior/'manifest.json')['sha256']==receipt['manifest_sha256'] and pin(prior/'result/result.json')['sha256']==receipt['result_sha256']
    assert pin(prior/'frozen.json')['sha256']==receipt['frozen_sha256']
    assert pin(corpus/'receipt.json')['sha256']==manifest['original_receipt_sha256']=='a4855ce825f4d5f1ea931440d031eb3a0aa5b3a0fee1fddbcf4d5b51500754b7'
    corpus_receipt=read(corpus/'receipt.json');native=read(corpus/'native-whisper/manifest.json')
    assert pin(corpus/'native-whisper/manifest.json')==corpus_receipt['files']['native-whisper/manifest.json']
    assert len(manifest['cases'])==len(native['cases'])==20 and [r['name'] for r in manifest['cases']]==[r['name'] for r in native['cases']]
    assert result['complete'] is True and result['core_sha256']==CORE and len(result['requests'])==42
    files={}
    def bind(path):
        files[path.relative_to(ROOT).as_posix()]=pin(path);return files[path.relative_to(ROOT).as_posix()]
    for path in [prior/'closed.json',prior/'receipt.json',prior/'manifest.json',prior/'result/result.json',prior/'frozen.json',corpus/'receipt.json',corpus/'native-whisper/manifest.json',ROOT/'tests/Shared/NpySupport.cs']:bind(path)
    def array(path,shape,expected_hash):
        assert pin(path)['sha256']==expected_hash;bind(path)
        value=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,dtype='<f4').reshape(shape)
        assert value.dtype==np.float32 and list(value.shape)==shape and np.isfinite(value).all()
        return dict(file=path.relative_to(ROOT).as_posix(),format='npy' if path.suffix=='.npy' else 'f32',shape=shape,raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    requests=[]
    for index in range(21):
        case=manifest['cases'][index%20];managed=result['requests'][2*index]
        assert managed['request']==index and managed['name']==case['name'] and managed['path']=='managed'
        mf,mh=managed['arrays'][:2];assert mf['shape']==[1,128,3000] and mh['shape']==[1,1500,1280]
        nf='native-whisper/'+case['features'];nh='native-whisper/'+case['name']+'-hidden.npy'
        assert pin(corpus/nf)==corpus_receipt['files'][nf]==manifest['files'][nf]
        assert pin(corpus/nh)==corpus_receipt['files'][nh]==manifest['files'][nh]
        requests.append(dict(request=index,name=case['name'],
            managed_features=array(prior/'result'/mf['file'],mf['shape'],mf['sha256']),managed_hidden=array(prior/'result'/mh['file'],mh['shape'],mh['sha256']),
            native_features=array(corpus/nf,[1,128,3000],case['features_sha256']),native_hidden=array(corpus/nh,[1,1500,1280],manifest['files'][nh]['sha256'])))
    assets=read(ROOT/'tests/whisper/transcription-assets.json');assert assets==manifest['assets']==native['assets']
    for name in ['onnx/encoder_model.onnx','onnx/encoder_model.onnx_data']:
        path=ROOT/'models/whisper-large-v3-turbo'/name;assert pin(path)==assets['files'][name];bind(path)
    for path in (base/'bin').iterdir():
        if path.is_file():bind(path)
    assert pin(base/'bin/Lokad.Onnx.dll')['sha256']==CORE
    for path in Path(__file__).parent.iterdir():
        if path.is_file():bind(path)
    interpreter=corpus/'venv/Scripts/python.exe'
    runtime=json.loads(subprocess.check_output([str(interpreter),'-X','utf8','-B',str(Path(__file__).with_name('native.py')),'--identity'],cwd=ROOT,text=True,encoding='utf-8'))
    assert (runtime['numpy'],runtime['onnxruntime'])==('2.2.4','1.29.0')
    spec=dict(schema=1,scope='Same-corpus encoder input/engine diagnostic; all saved baselines require exact bridges',core_sha256=CORE,
              model='models/whisper-large-v3-turbo/onnx/encoder_model.onnx',model_repository=assets['repository'],model_revision=assets['revision'],
              requests=requests,native_runtime=runtime,files=files,limits=dict(seconds=1800,rss=8*1024**3,available=1024**3,preflight_available=10*1024**3),
              cases=21,new_encoder_arrays=84,scaled_error_limit=1e-4)
    write(base/'manifest.json',spec);print('Frozen',len(files),'files;',pin(base/'manifest.json'))

if __name__=='__main__':main()
