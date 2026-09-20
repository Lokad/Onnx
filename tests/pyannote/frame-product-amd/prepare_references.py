"""Bind the observed AMD tables to the previously qualified double algorithms."""
from pathlib import Path
import sys, subprocess, shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tests/pyannote/filterbank-precision'))
from shared import *


def main():
    base=ROOT/'artifacts/wespeaker-frame-amd-reference-20260920';base.mkdir();(base/'inputs').mkdir()
    collected=ROOT/'artifacts/wespeaker-frame-product-amd-v4-20260920/collected'
    receipt=read(collected/'collection.json');assert receipt['terminal']
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    result=read(collected/'result/run.json');assert result['complete'] and result['code']==0
    proof=ROOT/'artifacts/wespeaker-precision-20260920';prior=read(proof/'manifest.json')
    assert pin(proof/'closed.json')['sha256']=='16aadd8e4e890c289a68f8bb0dccb738de1d87248396cc046f4b619177c05b0a'
    assert prior['interpreter']==pin(sys.executable)
    for path,wanted in prior['numeric'].items():assert pin(path)==wanted,path
    files={}
    def bind(path,wanted=None):
        value=pin(path)
        if wanted is not None:assert value==wanted,str(path)
        files[rel(path)]=value
    tables={}
    for name,stem,shape in [('Window','window',(400,)),('MelWeights','mel',(80,256))]:
        source=collected/'result/corpus'/(name+'.f32');bind(source)
        value=np.fromfile(source,dtype='<f4').reshape(shape);assert np.isfinite(value).all()
        path=base/'inputs'/(stem+'.npy')
        with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
        bind(path);tables[name]=dict(amd=pin(source),windows=pin(proof/'inputs'/(stem+'.f32')))
    for case in prior['cases']:bind(ROOT/case['input'],prior['files'][case['input']])
    for name in ['tests/pyannote/filterbank-precision/worker.py','tests/pyannote/filterbank-precision/shared.py',
                 'tests/pyannote/filterbank-precision/generate.py','tests/pyannote/filterbank-reference/common.py',
                 'tests/pyannote/filterbank-reference/routes.py']:bind(ROOT/name,prior['files'][name])
    for name in ['prepare_references.py','run_references.py']:bind(Path(__file__).with_name(name))
    shutil.copyfile(ROOT/'.agent/m3-filterbank-frame-product-20260920.md',base/'prospective-plan.md');bind(base/'prospective-plan.md')
    spec=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),cases=prior['cases'],files=files,
              window=rel(base/'inputs/window.npy'),mel=rel(base/'inputs/mel.npy'),numeric=prior['numeric'],interpreter=prior['interpreter'],
              tables=tables,limits=LIMITS,stages=STAGES,reference_limit=1e-8,original_limit=1e-4)
    write(base/'manifest.json',spec);print(json.dumps(dict(manifest=pin(base/'manifest.json'),files=len(files),cases=len(spec['cases']))))


if __name__=='__main__':main()
