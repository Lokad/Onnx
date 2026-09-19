"""Prepare matched Parakeet/Community-1 timing inputs from existing qualified artifacts."""
from pathlib import Path
import argparse,hashlib,json

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    def file(path,digest=None):
        path=Path(path);path=path if path.is_absolute() else root/path
        h=sha(path)
        if digest is not None and h!=digest:raise ValueError('Digest differs: '+str(path))
        return dict(path=path.relative_to(root).as_posix(),sha256=h,bytes=path.stat().st_size)
    asr=root/'artifacts/asr-labeled-20260919';receipt=read(asr/'receipt.json')
    assert sha(asr/'receipt.json')=='a4855ce825f4d5f1ea931440d031eb3a0aa5b3a0fee1fddbcf4d5b51500754b7'
    audio=read(asr/'inputs/audio.json');native=read(asr/'native-parakeet/manifest.json');assets=read(root/'tests/parakeet/transcribe/assets.json')
    for rel in ('inputs/audio.json','native-parakeet/manifest.json'):assert sha(asr/rel)==receipt['files'][rel]['sha256']
    models={name:file('models/parakeet-tdt-0.6b-v3/'+name,v['sha256']) for name,v in assets['files'].items()}
    cases=[]
    for c,n in zip(audio['cases'],native['cases'][:20],strict=True):
        assert c['name']==n['name'] and c['pcm_sha256']==n['pcm_sha256']
        cases.append(dict(name=c['name'],samples=c['samples'],pcm=file(asr/'inputs'/c['pcm'],c['pcm_sha256']),expected=n['expected']))
    upstream=root/'external/onnx-asr/src/onnx_asr/asr.py'
    assert hashlib.sha256(upstream.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==assets['reference']['asr_lf_sha256']
    parakeet=dict(schema=1,family='parakeet',models=models,graphs=assets['graphs'],cases=cases,assets=assets,
        upstream=file(upstream),reference=file(asr/'native-parakeet/manifest.json'),warmup_passes=1,measured_passes=3)
    prepared=root/'artifacts/pyannote-dialogue-20260919/archive/prepared';reference=read(prepared/'reference/manifest.json')
    pins=read(root/'tests/pyannote/dialogue/pins.json');assert reference['pins']==pins
    paths=dict(segmentation='models/pyannote-segmentation/segmentation/model.onnx',encoder='models/pyannote-embedding/embedding_encoder.onnx',
        projection='artifacts/wespeaker-api-20260919/reference/projection.onnx',plda='artifacts/pyannote-clustering-20260919/frozen/reference/prepared.json')
    models={name:file(path,pins['models'][name]) for name,path in paths.items()}
    native_assets={name:file('artifacts/pyannote-embedding-20260919/pipeline-config/plda/'+name,pins['assets'][name]) for name in ('plda.npz','xvec_transform.npz')}
    sources={name:file(prepared/'reference/upstream'/name) for name in pins['source_hashes']}
    for name,meta in sources.items():assert hashlib.sha256((root/meta['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pins['source_hashes'][name]
    cases=[]
    for c in reference['cases']:
        expected=dict(status=c['status'],windows=len(c['windows']),audio_seconds=c['seconds'],intervals=c['intervals'],exclusive_intervals=c['exclusive_intervals'],speakers=c.get('speakers',[]))
        cases.append(dict(name=c['name'],samples=round(c['seconds']*16000),pcm=file(prepared/'reference'/c['pcm'],reference['files'][c['pcm']]['sha256']),expected=expected))
    pyannote=dict(schema=1,family='pyannote',models=models,native_assets=native_assets,upstream=sources,pins=pins,cases=cases,
        reference=file(prepared/'reference/manifest.json'),warmup_passes=1,measured_passes=3)
    for family,record in [('parakeet',parakeet),('pyannote',pyannote)]:
        with (a.output/(family+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
        print(family,len(record['cases']),sum(c['samples'] for c in record['cases'])/16000,sha(a.output/(family+'.json')))

if __name__=='__main__':main()
