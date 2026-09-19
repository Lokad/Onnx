"""Fixed recording fixtures from preserved labeled PCM; no downloads or transcript-dependent selection."""
from pathlib import Path
import argparse,hashlib,json,struct
import numpy as np

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=json.loads(a.source.read_text(encoding='utf-8'));a.output.mkdir(parents=True,exist_ok=False)
    rows=[];arrays={}
    def add(name,pcm,label=None,maximum=4096,windows=256):
        pcm=np.ascontiguousarray(pcm,dtype=np.float32);assert pcm.ndim==1 and np.isfinite(pcm).all() and len(pcm)<=9600000
        path=a.output/(name+'.npy');np.save(path,pcm,allow_pickle=False);raw=pcm.astype('<f4',copy=False).tobytes()
        wave=a.output/(name+'.wav');wave.write_bytes(struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',36+len(raw),b'WAVE',b'fmt ',16,3,1,16000,64000,4,32,b'data',len(raw))+raw)
        arrays[name]=pcm;rows.append(dict(name=name,pcm=path.name,pcm_sha256=sha(path),wave=wave.name,wave_sha256=sha(wave),samples=len(pcm),
            reference_text=label,max_tokens=maximum,max_tokens_per_frame=10,max_windows=windows))
    for name in ('connected','shifted'):
        row=next(c for c in source['cases'] if c['name']==name);path=a.source.parent/row['pcm'];assert sha(path)==row['pcm_sha256']
        add(name,np.load(path,allow_pickle=False),row['reference_text'])
    # DC-offset stress forces hard boundaries; it is an artificial correlated case, not new natural speech.
    hard=arrays['connected']+np.float32(.02)
    assert all(np.mean(hard[i:i+160].astype(np.float64)**2)>.000009 for i in range(400000,480000,160))
    add('hard-boundary',hard,rows[0]['reference_text'])
    add('token-limit',arrays['connected'],maximum=2)
    add('window-limit',arrays['connected'],windows=1)
    add('maximum-speech',np.resize(arrays['connected'],9600000))
    add('tiny-tail',np.concatenate([np.full(480000,.02,np.float32),np.array([.1],np.float32)]))
    add('maximum-silence',np.zeros(9600000,np.float32))
    add('empty',np.zeros(0,np.float32))
    output=dict(schema=1,sample_rate=16000,scope='Constructed correlated boundary and finite resource fixtures; no natural long-audio qualification',
        source=str(a.source.resolve()),source_sha256=sha(a.source),selected=source['selected'],generator_sha256=sha(__file__),cases=rows)
    (a.output/'inputs.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print([(r['name'],r['samples']/16000) for r in rows])

if __name__=='__main__':main()
