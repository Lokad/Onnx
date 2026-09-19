"""Native split-model decisions with original pinned OpenAI timestamp/seek code.

Features are explicitly extracted per window. This is not OpenAI's complete
transcribe pipeline, its fallback/prompt policy, or a tensor qualification gate.
"""
from pathlib import Path
import argparse,ast,hashlib,inspect,json,time,types
import numpy as np
import torch
import onnxruntime as ort
import transformers
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from tokenizers import Tokenizer
from generate_rules import upstream,BEGIN,END,SIZE

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def seek_program(source):
    tree=ast.parse(source.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='transcribe')
    loop=next(n for n in ast.walk(function) if isinstance(n,ast.While) and ast.unparse(n.test)=='clip_idx < len(seek_clips)')
    first=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.AnnAssign) and ast.unparse(n.target)=='timestamp_tokens')
    nodes=loop.body[first:first+5]
    assert isinstance(nodes[-1],ast.If) and ast.unparse(nodes[-1].test)=='len(consecutive) > 0'
    return compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec')

def log_probability(scores,token):
    values=scores.astype(np.float64);maximum=values.max()
    return float(values[token]-maximum-np.log(np.exp(values-maximum).sum()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert (np.__version__,torch.__version__,ort.__version__,transformers.__version__)==('2.2.4','2.11.0+cpu','1.29.0','5.16.1')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    data=json.loads(a.inputs.read_text(encoding='utf-8'));source=a.inputs.parent/'upstream'
    pins=json.loads(Path(__file__).with_name('sources.json').read_text(encoding='utf-8'))
    assert data['source_revision']==pins['revision'] and data['sources']==pins['files']
    for name,item in data['sources'].items():assert sha(source/Path(name).name)==item['sha256']
    timestamp=upstream(source/'decoding.py');seek=seek_program(source/'transcribe.py')
    assets=json.loads((Path(__file__).parents[1]/'transcription-assets.json').read_text(encoding='utf-8'))
    for name,item in assets['files'].items():assert (a.models/name).stat().st_size==item['bytes'] and sha(a.models/name)==item['sha256'],name
    config=json.loads((a.models/'generation_config.json').read_text(encoding='utf-8'))
    tokenizer=Tokenizer.from_file(str(a.models/'tokenizer.json'))
    extractor=WhisperFeatureExtractor(feature_size=128,sampling_rate=16000,hop_length=160,chunk_length=30,n_fft=400,dither=0.0)
    extractor_sha=sha(Path(inspect.getfile(WhisperFeatureExtractor)))
    assert extractor_sha=='dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb'
    settings=ort.SessionOptions();settings.intra_op_num_threads=settings.inter_op_num_threads=1
    settings.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    settings.add_session_config_entry('session.intra_op.allow_spinning','0');settings.add_session_config_entry('session.inter_op.allow_spinning','0')
    sessions={name:ort.InferenceSession(str(a.models/'onnx'/filename),settings,providers=['CPUExecutionProvider'])
        for name,filename in [('encoder','encoder_model.onnx'),('first','decoder_model.onnx'),('past','decoder_with_past_model.onnx')]}
    files={};cases=[]
    def save_array(name,array):
        path=a.output/(name+'.npy');np.save(path,array,allow_pickle=False);files[path.name]=dict(sha256=sha(path),shape=list(array.shape),bytes=path.stat().st_size)
    def decode_window(pcm,case,name):
        if not np.any(pcm):return dict(text='',token_ids=[],stop_reason='SilentInput',skipped_as_no_speech=True,no_speech_probability=None,average_log_probability=None)
        padded=np.zeros((1,480000),np.float32);padded[0,:len(pcm)]=pcm
        features=extractor._np_extract_fbank_features(padded,'cpu');save_array(name+'-features',features)
        hidden=sessions['encoder'].run(None,{'input_features':features})[0];save_array(name+'-hidden',hidden)
        prefix=[config['decoder_start_token_id'],config['lang_to_id']['<|'+case['language']+'|>'],config['task_to_id']['transcribe']]
        tokens=[];cross={};previous={};total=0
        for step in range(case['max_new_tokens']):
            initial=step==0;session=sessions['first' if initial else 'past']
            feeds={'input_ids':np.array([prefix if initial else [tokens[-1]]],np.int64)}
            if initial:feeds['encoder_hidden_states']=hidden
            else:
                for inp in session.get_inputs():
                    if inp.name.startswith('past_key_values.'):
                        feeds[inp.name]=(cross if '.encoder.' in inp.name else previous)['present.'+inp.name[len('past_key_values.'):]]
            values=session.run(None,feeds);outputs={v.name:r for v,r in zip(session.get_outputs(),values)}
            logits=outputs['logits'];assert np.isfinite(logits).all();save_array(name+f'-{step:03d}-logits',logits)
            if initial:no_speech=float(np.exp(log_probability(logits[0,0],tokenizer.token_to_id('<|nospeech|>'))))
            scores=logits[0,-1].copy();scores[config['suppress_tokens']]=-np.inf
            if initial:scores[config['begin_suppress_tokens']]=-np.inf
            scores[END+1:BEGIN]=-np.inf
            timestamp.apply(torch.from_numpy(scores[None]),torch.tensor([prefix+tokens]))
            token=int(np.argmax(scores));assert np.isfinite(scores[token]);total+=log_probability(scores,token);tokens.append(token)
            if token==END:break
            if initial:cross=outputs
            previous=outputs
        ended=tokens[-1]==END;average=total/(len(tokens)-(1 if ended else 0)+1);skipped=no_speech>.6 and average<=-1
        return dict(text='' if skipped else tokenizer.decode([t for t in tokens if t<BEGIN],skip_special_tokens=True),token_ids=tokens,
            stop_reason='EndToken' if ended else 'TokenLimit',skipped_as_no_speech=skipped,no_speech_probability=no_speech,average_log_probability=average)
    def segments_for(result,position,length):
        limited=result['stop_reason']=='TokenLimit'
        if result['skipped_as_no_speech'] or result['stop_reason']=='SilentInput':return [],0 if limited else length
        tokens=result['token_ids'][:]
        if tokens and tokens[-1]==END:tokens.pop()
        raw=[]
        if limited:
            ends=[i for i in range(1,len(tokens)) if tokens[i-1]>=BEGIN and tokens[i]>=BEGIN]
            if len(tokens)>1 and tokens[-1]>=BEGIN and tokens[-2]<BEGIN:ends.append(len(tokens))
            prior=0;advance=0
            for end in ends:
                part=tokens[prior:end];raw.append(dict(start=(part[0]-BEGIN)*.02,end=(part[-1]-BEGIN)*.02,tokens=part));advance=(part[-1]-BEGIN)*320;prior=end
        else:
            scope=dict(torch=torch,tokens=torch.tensor(tokens),tokenizer=types.SimpleNamespace(timestamp_begin=BEGIN),time_offset=0.,time_precision=.02,
                segment_size=length,segment_duration=length/16000,input_stride=320,seek=0,result=result,current_segments=[],
                new_segment=lambda start,end,tokens,result:dict(start=start,end=end,tokens=tokens.tolist()))
            exec(seek,scope);raw=scope['current_segments'];advance=scope['seek']
        answer=[]
        for row in raw:
            start=min(length,round(row['start']*16000));end=min(length,round(row['end']*16000))
            ids=[t for t in row['tokens'] if t<END];text=tokenizer.decode(ids,skip_special_tokens=True)
            if end>start and text.strip():answer.append(dict(start_seconds=(position+start)/16000,end_seconds=(position+end)/16000,text=text,token_ids=ids))
        return answer,min(length,advance)
    for case in data['cases']:
        path=a.inputs.parent/case['pcm'];assert sha(path)==case['pcm_sha256'];pcm=np.load(path,allow_pickle=False);assert len(pcm)==case['samples']
        start=time.perf_counter();position=0;windows=[];segments=[];stop='Completed'
        while position<len(pcm):
            if len(windows)==case['max_windows']:stop='WindowLimit';break
            length=min(480000,len(pcm)-position);name=case['name']+f'-w{len(windows):03d}'
            result=decode_window(pcm[position:position+length],case,name);new,advance=segments_for(result,position,length);segments.extend(new)
            windows.append(dict(start_seconds=position/16000,audio_seconds=length/16000,advanced_seconds=advance/16000,decoding=result))
            position+=advance
            print(case['name'],len(windows),position/16000,result['stop_reason'],result['text'],flush=True)
            if result['stop_reason']=='TokenLimit':stop='TokenLimit';break
            if advance==0:stop='NoProgress';break
        recording=dict(text=tokenizer.decode([t for s in segments for t in s['token_ids']],skip_special_tokens=True),segments=segments,windows=windows,
                       stop_reason=stop,duration_seconds=len(pcm)/16000,processed_seconds=position/16000)
        cases.append(dict(name=case['name'],result=recording,seconds=time.perf_counter()-start,pcm_sha256=case['pcm_sha256']))
        (a.output/(case['name']+'.json')).write_text(json.dumps(cases[-1],indent=2)+'\n',encoding='utf-8')
    manifest=dict(schema=1,inputs_sha256=sha(a.inputs),assets=assets,source_files=data['sources'],generator_sha256=sha(Path(__file__)),
        timestamp_helper_sha256=sha(Path(__file__).with_name('generate_rules.py')),extractor_sha256=extractor_sha,
        numpy=np.__version__,onnxruntime=ort.__version__,torch=torch.__version__,transformers=transformers.__version__,cases=cases,files=files)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
