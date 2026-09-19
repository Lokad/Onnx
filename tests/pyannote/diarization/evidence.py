"""Identity and coverage checks shared by optional native generation and auditing."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
PINS=json.loads((HERE/'pins.json').read_text(encoding='utf-8'))
RECIPES=('pins.json','generate_reference.py','native_rules.py','evidence.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lfsha=lambda p:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def require(condition,message):
    if not condition:raise ValueError(message)

def recipes():return {name:lfsha(HERE/name) for name in RECIPES}

def local(directory,name):
    require(isinstance(name,str) and name==Path(name).name and '/' not in name and '\\' not in name,'Unsafe evidence path')
    return directory/name

def load(directory):
    m=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    require(m['pins']==PINS and m['recipes']==recipes(),'Reference recipe identity')
    require(m['models']==PINS['models'],'Reference model identity')
    require(set(m['files'])==set(PINS['layout']),'Reference file coverage')
    arrays={}
    for name,meta in m['files'].items():
        p=local(directory,name)
        require(sha(p)==meta['sha256'] and p.stat().st_size==meta['bytes'],'Reference digest: '+name)
        a=np.load(p,allow_pickle=False)
        require(dict(shape=list(a.shape),dtype=str(a.dtype))==PINS['layout'][name],'Reference layout: '+name)
        require(meta['shape']==list(a.shape) and meta['dtype']==str(a.dtype) and np.isfinite(a).all(),'Reference array: '+name)
        arrays[name]=a
    require([c['name'] for c in m['cases']]==[c['name'] for c in PINS['cases']],'Case coverage')
    referenced=set()
    for c,pin in zip(m['cases'],PINS['cases']):
        name=c['name']; pcm=c['pcm']
        require(pcm==name+'-pcm.npy' and sha(directory/pcm)==pin['pcm_sha256'],'PCM identity')
        require(arrays[pcm].shape==(pin['samples'],) and arrays[pcm].dtype==np.float32 and np.max(np.abs(arrays[pcm]))<=1,'PCM contract')
        require(c['seconds']==pin['samples']/16000 and len(c['windows'])==pin['windows'],'Recording geometry')
        referenced.add(pcm)
        for i,w in enumerate(c['windows']):
            expected={'scores','activity'} if name=='silence' else {'scores','activity','masks','features','encoded','pooled','vectors'}
            require(set(w)==expected,'Window stage coverage')
            for stage,file in w.items():
                require(file==f'{name}-{i}-{stage}.npy','Window reference pointer');referenced.add(file)
        for stage in ['count','centroids']+([] if name=='silence' else ['labels','original_labels','ordinary_native_frames','ordinary_frames','exclusive_native_frames','exclusive_frames']):
            file=c[stage];require(file==name+'-'+stage.replace('_','-')+'.npy','Case reference pointer');referenced.add(file)
        require(c['status']==('NoSpeech' if name=='silence' else 'Completed'),'Case status')
        for stage in ('intervals','exclusive_intervals'):
            for s,e,k in c[stage]:require(0<=s<e<=c['seconds'] and isinstance(k,int) and 0<=k<len(c.get('speakers',[])),'Reference interval')
    require(referenced==set(arrays),'Unreferenced or missing arrays')
    return m,arrays

def detail_keys(manifest):
    keys=[]
    for c in manifest['cases']:
        for w in c['windows']:
            for stage in ('scores','activity','masks','features','encoded','pooled','vectors'):
                if stage in w:keys.append((c['name'],stage,w[stage]))
        keys.append((c['name'],'count',c['count']))
        if 'labels' in c:keys.append((c['name'],'labels',c['labels']))
    return keys
