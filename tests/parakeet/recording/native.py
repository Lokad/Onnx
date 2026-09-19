"""Independent NumPy window planner and ORT decoder, crosschecked against pinned original onnx-asr."""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,re,time,types
import numpy as np
import onnxruntime as ort

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def windows(pcm):
    """Vectorized energy/run endpoints; independent of the managed incremental scan."""
    start=0
    while start<len(pcm):
        count=min(480000,len(pcm)-start);boundary='EndOfRecording'
        if start+count<len(pcm):
            energy=(pcm[start+400000:start+480000].astype(np.float64).reshape(-1,160)**2).sum(axis=1)
            changes=np.diff(np.r_[False,energy<=160*.000009,False].astype(np.int8))
            beginnings=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
            runs=[(int(e-b),int(b)) for b,e in zip(beginnings,ends) if e-b>=20]
            if runs:
                blocks,first=max(runs);count=400000+first*160+blocks*80;boundary='Quiet'
            else:boundary='HardLimit'
        yield start,count,boundary
        start+=count

class Native:
    def __init__(self,models,source):
        helper=load('parakeet_original',Path(__file__).parents[1]/'transcribe/generate_reference.py')
        self.assets=json.loads((Path(__file__).parents[1]/'transcribe/assets.json').read_text(encoding='utf-8'))
        for name,pin in self.assets['files'].items():assert (models/name).stat().st_size==pin['bytes'] and sha(models/name)==pin['sha256'],name
        assert helper.source_sha(source)==self.assets['reference']['asr_lf_sha256']
        ns=dict(np=np,re=re,TimestampedResult=collections.namedtuple('TimestampedResult','text timestamps tokens logprobs'))
        self.loop=helper.upstream_method(source.read_text(encoding='utf-8'),'_AsrWithTransducerDecoding','_decoding',ns)
        self.text=helper.upstream_method(source.read_text(encoding='utf-8'),'_AsrWithDecoding','_decode_tokens',ns)
        self.vocab={}
        for line in (models/'vocab.txt').read_text(encoding='utf-8').splitlines():
            piece,index=line.rsplit(' ',1);self.vocab[int(index)]=piece.replace('\u2581',' ')
        options=ort.SessionOptions();options.log_severity_level=4;options.intra_op_num_threads=options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for setting in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):options.add_session_config_entry(setting,'0')
        self.sessions={k:ort.InferenceSession(str(models/v),options,providers=['CPUExecutionProvider']) for k,v in self.assets['graphs'].items()}
        assert all(s.get_providers()==['CPUExecutionProvider'] for s in self.sessions.values())
        self.crosschecks=0
    def graph(self,name,feeds):
        before={k:v.tobytes() for k,v in feeds.items()};s=self.sessions[name];values=s.run(None,feeds)
        assert all(np.isfinite(v).all() for v in values) and all(v.tobytes()==before[k] for k,v in feeds.items())
        return dict(zip([v.name for v in s.get_outputs()],values))
    def decode(self,pcm,maximum,per_frame):
        if not np.any(pcm):return dict(text='',token_ids=[],frame_indices=[],duration_frames=[],stop_reason='SilentInput',encoded_frames=0,decoder_calls=0)
        if len(pcm)<257:pcm=np.pad(pcm,(0,257-len(pcm)))
        pre=self.graph('frontend',dict(waveforms=pcm[None,:],waveforms_lens=np.array([len(pcm)],np.int64)))
        enc=self.graph('encoder',dict(audio_signal=pre['features'],length=pre['features_lens']));hidden=enc['outputs'];frames=int(enc['encoded_lengths'][0])
        assert hidden.shape==(1,1024,frames) and frames==(len(pcm)//160+8)//8
        def zero():return np.zeros((2,1,640),np.float32),np.zeros((2,1,640),np.float32)
        def step(previous,states,vector):
            return self.graph('decoder',dict(encoder_outputs=np.ascontiguousarray(vector[None,:,None]),
                targets=np.array([[previous[-1] if previous else 8192]],np.int32),target_length=np.array([1],np.int32),input_states_1=states[0],input_states_2=states[1]))
        state=zero();tokens=[];positions=[];durations=[];frame=emitted=calls=0
        while frame<frames and len(tokens)<maximum:
            out=step(tokens,state,hidden[0,:,frame]);assert out['outputs'].shape==(1,1,1,8198) and np.array_equal(out['prednet_lengths'],[1])
            logits=out['outputs'].reshape(-1);token=int(logits[:8193].argmax());duration=int(logits[8193:].argmax());calls+=1
            if token!=8192:
                tokens.append(token);positions.append(frame);durations.append(duration);state=out['output_states_1'],out['output_states_2'];emitted+=1
            if duration:frame+=duration;emitted=0
            elif token==8192 or emitted==per_frame:frame+=1;emitted=0
        def original_step(previous,states,vector):
            out=step(previous,states,vector);logits=out['outputs'].reshape(-1)
            return logits[:8193],int(logits[8193:].argmax()),(out['output_states_1'],out['output_states_2'])
        shim=types.SimpleNamespace(use_low_precision=False,_blank_idx=8192,_vocab_size=8193,_max_tokens_per_step=per_frame,_create_state=zero,_decode=original_step,
            _vocab=self.vocab,DECODE_SPACE_PATTERN=re.compile(r'\A\s|\s\B|(\s)\b'),window_step=.01,_subsampling_factor=8)
        original_tokens,original_frames,_=next(self.loop(shim,hidden.transpose(0,2,1),np.array([frames],np.int64)))
        assert tokens==original_tokens[:len(tokens)] and positions==original_frames[:len(tokens)] and (frame<frames or tokens==original_tokens)
        self.crosschecks+=1
        return dict(text=self.text(shim,tokens,positions,None).text,token_ids=tokens,frame_indices=positions,duration_frames=durations,
            stop_reason='EndOfAudio' if frame>=frames else 'TokenLimit',encoded_frames=frames,decoder_calls=calls)
    def recording(self,pcm,case):
        result=[];text=[];processed=0;stop='Completed'
        for start,length,boundary in windows(pcm):
            if len(result)==case['max_windows']:stop='WindowLimit';break
            decoded=self.decode(pcm[start:start+length],case['max_tokens'],case['max_tokens_per_frame'])
            result.append(dict(start_seconds=start/16000,audio_seconds=length/16000,boundary=boundary,decoding=decoded))
            if decoded['stop_reason']=='TokenLimit':stop='TokenLimit';break
            processed=start+length
            if decoded['text']:text.append(decoded['text'])
        return dict(text=' '.join(text),windows=result,stop_reason=stop,duration_seconds=len(pcm)/16000,processed_seconds=processed/16000)

def main():
    p=argparse.ArgumentParser();
    for n in ('models','inputs','source','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert (np.__version__,ort.__version__)==('2.2.4','1.29.0');a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.inputs.read_text(encoding='utf-8'));model=Native(a.models,a.source);rows=[]
    for index,case in enumerate(manifest['cases']+manifest['cases'][:1]):
        path=a.inputs.parent/case['pcm'];assert sha(path)==case['pcm_sha256'];pcm=np.load(path,allow_pickle=False);before=pcm.tobytes()
        t=time.monotonic();result=model.recording(pcm,case);assert pcm.tobytes()==before
        rows.append(dict(name=case['name'],repeat=index==len(manifest['cases']),result=result,seconds=time.monotonic()-t))
        (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
        print(case['name'],result['stop_reason'],len(result['windows']),rows[-1]['seconds'],flush=True)
    assert rows[0]['result']==rows[-1]['result']
    output=dict(schema=1,inputs_sha256=sha(a.inputs),source_sha256=sha(a.source),generator_sha256=sha(__file__),assets=model.assets,
        numpy=np.__version__,ort=ort.__version__,upstream_crosschecks=model.crosschecks,cases=rows)
    (a.output/'result.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
