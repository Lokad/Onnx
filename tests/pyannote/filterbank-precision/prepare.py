"""Freeze 53 complete existing cases and source-derived arithmetic variants."""
import argparse, shutil, subprocess
from shared import *
from generate import generate


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);a=p.parse_args();base=Path(a.artifact).resolve()
    base.mkdir(parents=True,exist_ok=False);(base/'inputs').mkdir();(base/'app').mkdir();files={}
    psutil_module().Process().cpu_affinity([0])
    def bind(path,wanted=None):
        name=rel(path)
        if name not in files:files[name]=pin(path)
        if wanted is not None:assert files[name]==wanted,name
        return files[name]
    manifests={}
    for name,digest in PRIORS.items():
        old=ROOT/'artifacts'/name;assert bind(old/'closed.json')['sha256']==digest
        receipt=read(old/'closed.json');prior=read(old/'manifest.json');manifests[name]=prior
        for path,wanted in receipt['files'].items():bind(old/path,wanted)
        for path,wanted in receipt['reports'].items():bind(ROOT/path,wanted)
        for path,wanted in prior['files'].items():bind(ROOT/path,wanted)
        assert prior['interpreter']==pin(sys.executable)
        for path,wanted in prior['numeric'].items():assert pin(path)==wanted
    captured=ROOT/'artifacts/wespeaker-coefficients-20260920'
    for name,count in [('Window',400),('MelWeights',20480)]:
        assert pin(captured/'five'/(name+'.f32'))==pin(captured/'dialogue'/(name+'.f32'))
        raw=np.fromfile(captured/'dialogue'/(name+'.f32'),dtype='<f4');assert raw.size==count and np.isfinite(raw).all()
        stem='window' if name=='Window' else 'mel';shutil.copyfile(captured/'dialogue'/(name+'.f32'),base/'inputs'/(stem+'.f32'))
        with (base/'inputs'/(stem+'.npy')).open('xb') as stream:np.save(stream,raw if name=='Window' else raw.reshape(80,256),allow_pickle=False)
    cases=[]
    initial=manifests['wespeaker-full-reference-20260920'];windows=manifests['wespeaker-window-reference-20260920']
    for prior_case in initial['cases']:
        shape=prior_case['shape'];baselines={}
        for key,field,kind in [('native','native','npy'),('windows-default','managed','raw')]:
            path=ROOT/prior_case[field];baselines[key]=dict(file=rel(path),pin=pin(path),shape=shape,format=kind)
        cases.append(dict(name='unit-'+prior_case['name'],corpus='unit',input=prior_case['input'],baselines=baselines,
                          old_reference='artifacts/wespeaker-full-reference-20260920',old_name=prior_case['name']))
    for prior_case in windows['cases']:
        cases.append(dict(name=prior_case['name'],corpus=prior_case['corpus'],input=prior_case['input'],baselines=prior_case['baselines'],
                          old_reference='artifacts/wespeaker-window-reference-20260920',old_name=prior_case['name']))
    assert len(cases)==53 and len({c['name'] for c in cases})==53
    for case in cases:
        pcm=np.load(ROOT/case['input'],allow_pickle=False);assert pcm.dtype==np.float32 and pcm.ndim==1 and np.isfinite(pcm).all() and np.max(np.abs(pcm))<=1
        case.update(samples=int(pcm.size),shapes=shapes(int(pcm.size)))
        with (base/'inputs'/(case['name']+'.f32')).open('xb') as f:f.write(pcm.tobytes())
        for desc in case['baselines'].values():load_baseline(desc)
    folder=Path(__file__).resolve().parent
    subprocess.run(['git','diff','--exit-code','HEAD','--',str(folder)],cwd=ROOT,check=True)
    assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',str(folder)],cwd=ROOT).strip()
    for path in folder.iterdir():
        if path.is_file():bind(path)
    assert bind(PRODUCT)['sha256']==SOURCE_SHA and bind(CORE)['sha256']==CORE_SHA
    for name in ['Program.cs','Precision.csproj']:shutil.copyfile(folder/name,base/'app'/name)
    product=PRODUCT.read_bytes()
    for variant in VARIANTS:
        with (base/'app'/(variant+'.cs')).open('x',encoding='utf-8') as f:f.write(generate(product,variant))
    shutil.copyfile(ROOT/'.agent/m3-filterbank-precision-20260920.md',base/'protocol.md')
    tests=subprocess.run([sys.executable,'-X','utf8','-B',str(folder/'test_precision.py')],capture_output=True,text=True,timeout=60)
    write(base/'tests.json',dict(code=tests.returncode,stdout=tests.stdout,stderr=tests.stderr));assert tests.returncode==0
    for path in sorted(base.rglob('*')):
        if path.is_file():bind(path)
    spec=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=files,cases=cases,
              window=rel(base/'inputs/window.npy'),mel=rel(base/'inputs/mel.npy'),variants=VARIANTS,stages=STAGES,limits=LIMITS,
              numeric=initial['numeric'],interpreter=initial['interpreter'],reference_limit=REFERENCE_LIMIT,original_limit=ORIGINAL_LIMIT,
              rounded_limit=1e-6,core=rel(CORE),tests=pin(base/'tests.json'))
    write(base/'manifest.json',spec);print(json.dumps(dict(cases=len(cases),files=len(files),manifest=pin(base/'manifest.json'))))


if __name__=='__main__':main()
