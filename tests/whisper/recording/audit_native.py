"""Recompute every native greedy choice/confidence from retained full raw logits."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import torch
from generate_rules import upstream,BEGIN,END,SIZE

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def probability(logits,index):
    values=logits.astype(np.float64);maximum=float(np.max(values))
    return float(values[index])-maximum-math.log(math.fsum(math.exp(float(v)-maximum) for v in values))

def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    manifest=json.loads(a.native.read_text());inputs=json.loads(a.inputs.read_text());config=json.loads((a.models/'generation_config.json').read_text())
    assert manifest['inputs_sha256']==sha(a.inputs)
    assert sha(a.models/'generation_config.json')==manifest['assets']['files']['generation_config.json']['sha256']
    rule=upstream(a.inputs.parent/'upstream/decoding.py');steps=0;rows=[]
    for case,parameters in zip(manifest['cases'],inputs['cases'],strict=True):
        assert case['name']==parameters['name']
        prefix=[config['decoder_start_token_id'],config['lang_to_id']['<|'+parameters['language']+'|>'],config['task_to_id']['transcribe']]
        for window_index,window in enumerate(case['result']['windows']):
            d=window['decoding']
            if d['stop_reason']=='SilentInput':continue
            tokens=[];total=0;prefix_name=case['name']+f'-w{window_index:03d}'
            for step,expected in enumerate(d['token_ids']):
                name=prefix_name+f'-{step:03d}-logits.npy';path=a.native.parent/name
                assert sha(path)==manifest['files'][name]['sha256']
                values=np.load(path,allow_pickle=False);assert values.dtype==np.float32 and values.shape==(1,3 if step==0 else 1,SIZE) and np.isfinite(values).all()
                if step==0:no_speech=math.exp(probability(values[0,0],50363))
                scores=values[0,-1].copy();scores[config['suppress_tokens']]=-np.inf
                if step==0:scores[config['begin_suppress_tokens']]=-np.inf
                scores[END+1:BEGIN]=-np.inf
                rule.apply(torch.from_numpy(scores[None]),torch.tensor([prefix+tokens]))
                chosen=int(np.argmax(scores));assert chosen==expected and np.isfinite(scores[chosen]),(prefix_name,step,chosen,expected)
                total+=probability(scores,chosen);tokens.append(chosen);steps+=1
            denominator=len(tokens)+(0 if tokens[-1]==END else 1);average=total/denominator
            assert abs(no_speech-d['no_speech_probability'])<=1e-12 and abs(average-d['average_log_probability'])<=1e-12
            rows.append(dict(case=case['name'],window=window_index,steps=len(tokens),no_speech_error=abs(no_speech-d['no_speech_probability']),average_error=abs(average-d['average_log_probability'])))
    result=dict(schema=1,native_sha256=sha(a.native),inputs_sha256=sha(a.inputs),auditor_sha256=sha(Path(__file__)),
                timestamp_helper_sha256=sha(Path(__file__).with_name('generate_rules.py')),steps=steps,windows=rows,passed=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(steps,'native choices independently reconstructed')

if __name__=='__main__':main()
