"""Independent native encoder/greedy decoder oracle from hash-bound PCM/features.

The feature manifest records the independent NumPy frontend, audio source and
resampling recipe. The managed replay must extract features from PCM itself.
This lane checks application agreement; it does not replace tensor-error gates.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--audio',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert (np.__version__,ort.__version__)==('2.2.4','1.29.0')
    assets=json.loads((Path(__file__).parent/'transcription-assets.json').read_text(encoding='utf-8'))
    for name,item in assets['files'].items():
        path=args.models/name;assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
    audio=json.loads(args.audio.read_text(encoding='utf-8'));assert audio['sample_rate']==16000
    generation=json.loads((args.models/'generation_config.json').read_text(encoding='utf-8'))
    tokenizer=Tokenizer.from_file(str(args.models/'tokenizer.json'))
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    options.add_session_config_entry('session.inter_op.allow_spinning','0')
    sessions={name:ort.InferenceSession(str(args.models/'onnx'/file),options,providers=['CPUExecutionProvider']) for name,file in
        [('encoder','encoder_model.onnx'),('first','decoder_model.onnx'),('past','decoder_with_past_model.onnx')]}
    eos=generation['eos_token_id'];nospeech=tokenizer.token_to_id('<|nospeech|>')
    def log_probability(values,token):
        values=values.astype(np.float64);m=values.max()
        return float(values[token]-m-np.log(np.exp(values-m).sum()))
    cases=[]
    for case in audio['cases']:
        start=time.perf_counter()
        for kind in ('pcm','features'):
            path=args.audio.parent/case[kind];assert sha(path)==case[kind+'_sha256']
            shutil.copyfile(path,args.output/case[kind])
        pcm=np.load(args.output/case['pcm'],allow_pickle=False)
        assert np.isfinite(pcm).all()
        if np.count_nonzero(pcm)==0:
            cases.append(dict(**case,tokens=[],text='',raw_text='',stopped_on_eos=False,stop_reason='SilentInput',
                no_speech_probability=None,average_log_probability=None,skipped_as_no_speech=True,steps=[],seconds=0))
            print(case['name'],'exact digital silence: no inference',flush=True)
            continue
        features=np.load(args.output/case['features'],allow_pickle=False)
        hidden=sessions['encoder'].run(None,{'input_features':features})[0]
        np.save(args.output/(case['name']+'-hidden.npy'),hidden,allow_pickle=False)
        prefix=[generation['decoder_start_token_id'],generation['lang_to_id']['<|'+case['language']+'|>'],generation['task_to_id']['transcribe'],generation['no_timestamps_token_id']]
        tokens=[];cross={};previous={};total=0;steps=[]
        for step in range(case['max_new_tokens']):
            session=sessions['first' if step==0 else 'past']
            feeds={'input_ids':np.array([prefix if step==0 else [tokens[-1]]],np.int64)}
            if step==0:feeds['encoder_hidden_states']=hidden
            else:
                for inp in session.get_inputs():
                    if inp.name.startswith('past_key_values.'):
                        feeds[inp.name]=(cross if '.encoder.' in inp.name else previous)['present.'+inp.name[len('past_key_values.'):]]
            values=session.run(None,feeds);outputs={d.name:v for d,v in zip(session.get_outputs(),values)}
            logits=outputs['logits'];assert np.isfinite(logits).all()
            if step==0:no_speech=float(np.exp(log_probability(logits[0,0],nospeech)))
            scores=logits[0,-1].copy();scores[generation['suppress_tokens']]=-np.inf
            if step==0:scores[generation['begin_suppress_tokens']]=-np.inf
            scores[eos+1:]=-np.inf
            token=int(np.argmax(scores));total+=log_probability(scores,token);tokens.append(token)
            file=args.output/f"{case['name']}-{step:03d}-logits.npy";np.save(file,logits,allow_pickle=False)
            steps.append(dict(token=token,logits=file.name,sha256=sha(file)))
            if token==eos:break
            if step==0:cross=outputs
            previous=outputs
        stopped=tokens[-1]==eos;count=len(tokens)-(1 if stopped else 0);average=total/(count+1)
        skipped=no_speech>0.6 and average<=-1.0
        text=tokenizer.decode(tokens,skip_special_tokens=True)
        row=dict(**case,tokens=tokens,text='' if skipped else text,raw_text=text,stopped_on_eos=stopped,no_speech_probability=no_speech,
            average_log_probability=average,skipped_as_no_speech=skipped,stop_reason='EndToken' if stopped else 'TokenLimit',steps=steps,seconds=time.perf_counter()-start)
        cases.append(row);print(case['name'],row['text'],tokens,'no speech',no_speech,'average',average,flush=True)
        (args.output/'manifest.json').write_text(json.dumps(dict(schema=1,assets=assets,onnxruntime=ort.__version__,numpy=np.__version__,
            audio_manifest_sha256=sha(args.audio),generator_sha256=sha(Path(__file__)),cases=cases),indent=2),encoding='utf-8')
    (args.output/'manifest.json').write_text(json.dumps(dict(schema=1,assets=assets,onnxruntime=ort.__version__,numpy=np.__version__,
        audio_manifest_sha256=sha(args.audio),generator_sha256=sha(Path(__file__)),cases=cases),indent=2),encoding='utf-8')

if __name__=='__main__':main()
