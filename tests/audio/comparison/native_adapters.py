"""Complete PCM applications around CPU ORT graphs, without reference-export work."""
from pathlib import Path
import ast,collections,functools,importlib.util,json,re,types
import numpy as np
import onnxruntime as ort

def session(path):
    options=ort.SessionOptions();options.log_severity_level=4
    options.intra_op_num_threads=options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for name in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):options.add_session_config_entry(name,'0')
    result=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    assert result.get_providers()==['CPUExecutionProvider']
    return result

def original_method(path,cls,name,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    container=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    method=next(n for n in container.body if isinstance(n,ast.FunctionDef) and n.name==name)
    code=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(path),'exec'),namespace)
    return namespace[name]

class Parakeet:
    def __init__(self,root,manifest):
        paths={name:root/info['path'] for name,info in manifest['models'].items()}
        self.sessions={name:session(paths[file]) for name,file in manifest['graphs'].items()}
        self.outputs={name:[v.name for v in s.get_outputs()] for name,s in self.sessions.items()}
        vocab={}
        for line in paths['vocab.txt'].read_text(encoding='utf-8').splitlines():
            piece,index=line.rsplit(' ',1);vocab[int(index)]=piece.replace('\u2581',' ')
        assert len(vocab)==8193 and vocab[8192]=='<blk>'
        namespace=dict(np=np,re=re,TimestampedResult=collections.namedtuple('TimestampedResult','text timestamps tokens logprobs'))
        self.decode_text=original_method(root/manifest['upstream']['path'],'_AsrWithDecoding','_decode_tokens',namespace)
        self.shim=types.SimpleNamespace(_vocab=vocab,DECODE_SPACE_PATTERN=re.compile(r'\A\s|\s\B|(\s)\b'),window_step=.01,_subsampling_factor=8)

    def graph(self,name,feeds):return dict(zip(self.outputs[name],self.sessions[name].run(None,feeds)))

    def __call__(self,pcm):
        prepared=self.graph('frontend',dict(waveforms=pcm[None,:],waveforms_lens=np.array([len(pcm)],np.int64)))
        encoded=self.graph('encoder',dict(audio_signal=prepared['features'],length=prepared['features_lens']))
        hidden=encoded['outputs'];frames=int(encoded['encoded_lengths'][0]);state=(np.zeros((2,1,640),np.float32),np.zeros((2,1,640),np.float32))
        tokens=[];positions=[];durations=[];frame=emitted=calls=0
        while frame<frames and len(tokens)<4096:
            values=self.graph('decoder',dict(encoder_outputs=np.ascontiguousarray(hidden[0,:,frame][None,:,None]),
                targets=np.array([[tokens[-1] if tokens else 8192]],np.int32),target_length=np.array([1],np.int32),input_states_1=state[0],input_states_2=state[1]))
            logits=values['outputs'].reshape(-1);token,duration=int(logits[:8193].argmax()),int(logits[8193:].argmax());calls+=1
            if token!=8192:
                tokens.append(token);positions.append(frame);durations.append(duration);state=values['output_states_1'],values['output_states_2'];emitted+=1
            if duration:frame+=duration;emitted=0
            elif token==8192 or emitted==10:frame+=1;emitted=0
        return dict(text=self.decode_text(self.shim,tokens,positions,None).text,token_ids=tokens,frame_indices=positions,duration_frames=durations,
            stop_reason='EndOfAudio' if frame>=frames else 'TokenLimit',encoded_frames=frames,decoder_calls=calls)

class StableNumpy:
    def __getattr__(self,name):return getattr(np,name)
    def argsort(self,a,axis=-1):return np.argsort(a,axis=axis,kind='stable')

class Pyannote:
    def __init__(self,root,manifest):
        import torch
        from torchaudio.compliance import kaldi
        from scipy.cluster.hierarchy import linkage,fcluster
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
        from einops import rearrange
        self.torch=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.get_num_threads()==torch.get_num_interop_threads()==1
        paths={name:root/info['path'] for name,info in manifest['models'].items()}
        self.seg=session(paths['segmentation']);self.enc=session(paths['encoder']);self.proj=session(paths['projection'])
        upstream=(root/manifest['upstream']['pipelines/clustering.py']['path']).parents[1]
        spec=importlib.util.spec_from_file_location('benchmark_pyannote_rules',root/'tests/pyannote/diarization/native_rules.py')
        rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules);rules.configure(upstream);self.rules=rules
        self.compute_fbank=original_method(upstream/'models/embedding/wespeaker/__init__.py','BaseWeSpeakerResNet','compute_fbank',dict(torch=torch))
        settings=dict(num_mel_bins=80,frame_length=25.,frame_shift=10.,round_to_power_of_two=True,snip_edges=True,dither=0.,sample_frequency=16000,window_type='hamming',use_energy=False)
        self.frontend=types.SimpleNamespace(hparams=types.SimpleNamespace(fbank_centering_span=None),_fbank=functools.partial(kaldi.fbank,**settings))
        namespace={};path=upstream/'models/blocks/pooling.py';exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace);self.pool=namespace['StatsPool']()
        namespace={};path=upstream/'utils/vbx.py';exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace)
        assets=manifest['native_assets'];tf,plda,phi=namespace['vbx_setup'](root/assets['xvec_transform.npz']['path'],root/assets['plda.npz']['path'])
        class Plda:
            def __init__(self):self.phi=phi
            def __call__(self,data):return plda(tf(data))
        cluster=types.SimpleNamespace(plda=Plda(),threshold=.6,Fa=.07,Fb=.8,metric='cosine',constrained_assignment=True)
        cluster_namespace=dict(np=np,linkage=linkage,fcluster=fcluster,linear_sum_assignment=linear_sum_assignment,cdist=cdist,rearrange=rearrange,cluster_vbx=namespace['cluster_vbx'])
        path=upstream/'pipelines/clustering.py'
        for method in ('filter_embeddings','constrained_argmax'):
            setattr(cluster,method,types.MethodType(original_method(path,'BaseClustering',method,cluster_namespace),cluster))
        self.cluster=types.MethodType(original_method(path,'VBxClustering','__call__',cluster_namespace),cluster)

    def __call__(self,pcm):
        torch=self.torch;rules=self.rules;env=rules.env
        waveform=rules.windows(pcm);activity=np.stack([rules.powerset(self.seg.run(None,{'waveform':wave[None]})[0])[0] for wave in waveform])
        counts=env['SpeakerDiarizationMixin'].speaker_count(rules.feature(activity),rules.frames,warm_up=(0.,0.))
        if counts.data.max()==0:return dict(status='NoSpeech',windows=len(waveform),audio_seconds=len(pcm)/16000,intervals=[],exclusive_intervals=[],speakers=[])
        masks,_=rules.masks(activity,pcm);vectors=[]
        for i,wave in enumerate(waveform):
            features=self.compute_fbank(self.frontend,torch.from_numpy(wave[None])).numpy()
            encoded=self.enc.run(None,{'fbank_features':features})[0];pooled=[]
            for mask in masks[i*3:i*3+3]:
                stats=self.pool(torch.from_numpy(encoded),torch.from_numpy(mask[None])).numpy()
                pooled.append(self.proj.run(None,{'pooled':stats})[0][0])
            vectors.append(np.stack(pooled))
        hard,_,centers=self.cluster(embeddings=np.stack(vectors),segmentations=types.SimpleNamespace(data=activity),min_clusters=1,max_clusters=np.inf)
        hard[~activity.any(axis=1)]=-2;order={}
        for label in hard.reshape(-1):
            if label>=0 and int(label) not in order:order[int(label)]=len(order)
        for label in range(len(centers)):
            if label not in order:order[label]=len(order)
        canonical=np.asarray([order.get(int(label),-2) for label in hard.reshape(-1)]).reshape(hard.shape)
        ordered=np.empty_like(centers)
        for old,new in order.items():ordered[new]=centers[old]
        timelines={};env['np']=StableNumpy()
        try:
            for kind,cap in [('intervals',127),('exclusive_intervals',1)]:
                count=env['SlidingWindowFeature'](np.minimum(counts.data,cap).astype(np.int8),counts.sliding_window)
                binary=env['SpeakerDiarization']().reconstruct(rules.feature(activity),canonical,count)
                timelines[kind]=[[max(0.,s),min(len(pcm)/16000,e),k] for s,e,k in rules.intervals(binary) if min(len(pcm)/16000,e)>max(0.,s)]
        finally:env['np']=np
        visible=sorted(set(v[2] for v in timelines['intervals']));mapping={old:new for new,old in enumerate(visible)}
        return dict(status='Completed' if visible else 'NoSpeech',windows=len(waveform),audio_seconds=len(pcm)/16000,
            intervals=[[s,e,mapping[k]] for s,e,k in timelines['intervals']],exclusive_intervals=[[s,e,mapping[k]] for s,e,k in timelines['exclusive_intervals']],
            speakers=[dict(speaker=mapping[k],centroid=ordered[k].tolist() if k<len(ordered) else [0.]*256,has_embedding=k<len(ordered)) for k in visible])
