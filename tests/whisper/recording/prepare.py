"""Prepare fixed constructed recordings from the already qualified labeled subset; no model download."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
import numpy as np
import soundfile as sf

REVISION='86098128c0b4f24f0e2aa2994de830614b474227'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--audio',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=a.output/'upstream';source.mkdir()
    sources={}
    pins=json.loads(Path(__file__).with_name('sources.json').read_text(encoding='utf-8'))
    for name in ('whisper/decoding.py','whisper/transcribe.py','LICENSE'):
        path=source/Path(name).name
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/openai/whisper/{REVISION}/{name}',timeout=60) as response:path.write_bytes(response.read())
        sources[name]=dict(sha256=sha(path),bytes=path.stat().st_size)
        assert sources[name]==pins['files'][name],name
    manifest=json.loads(a.audio.read_text(encoding='utf-8'));pieces=[];selected=[]
    for item in manifest['cases'][:6]:
        path=a.audio.parent/item['pcm'];assert sha(path)==item['pcm_sha256']
        samples=np.load(path,allow_pickle=False);assert samples.dtype==np.float32 and samples.ndim==1 and np.isfinite(samples).all()
        selected.append(dict(name=item['name'],samples=len(samples),pcm_sha256=sha(path),reference_text=item['reference_text']))
        if pieces:pieces.append(np.zeros(4800,np.float32))
        pieces.append(samples)
    connected=np.concatenate(pieces);shifted=np.concatenate([np.zeros(37920,np.float32),connected])
    rows=[]
    for name,pcm in [('connected',connected),('shifted',shifted)]:
        path=a.output/(name+'.npy');np.save(path,pcm,allow_pickle=False)
        wave=a.output/(name+'.wav');sf.write(wave,pcm,16000,subtype='FLOAT')
        assert np.array_equal(sf.read(wave,dtype='float32')[0],pcm)
        rows.append(dict(name=name,pcm=path.name,pcm_sha256=sha(path),wave=wave.name,wave_sha256=sha(wave),samples=len(pcm),
                         reference_text=' '.join(v['reference_text'] for v in selected),language='en',max_new_tokens=444,max_windows=256))
    for name,base,token_limit,window_limit in [('token-limit','connected',2,256),('window-limit','connected',444,1)]:
        row=dict(next(r for r in rows if r['name']==base));row.update(name=name,max_new_tokens=token_limit,max_windows=window_limit);rows.append(row)
    output=dict(schema=1,scope='Six fixed read-speech utterances, joined with 0.3 seconds silence, plus shifted boundary diagnostic',
        source_manifest_sha256=sha(a.audio),source_revision=REVISION,sources=sources,selected=selected,sample_rate=16000,
        generator_sha256=sha(Path(__file__)),cases=rows)
    (a.output/'inputs.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print([(r['name'],r['samples']/16000) for r in rows])

if __name__=='__main__':main()
