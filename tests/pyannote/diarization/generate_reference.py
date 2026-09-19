"""Generate pinned native Community-1 pipeline references using local assets only."""
from pathlib import Path
import argparse,ast,functools,hashlib,importlib.metadata,inspect,json,subprocess,types
import numpy as np
import torch
import onnxruntime as ort
from torchaudio.compliance import kaldi
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from einops import rearrange
from evidence import sha,lfsha,recipes,PINS
from native_rules import configure,env,extract,windows,powerset,masks,timeline

parser=argparse.ArgumentParser(description=__doc__)
for name in ('reference-source','segmentation','embedding-directory','projection','prepared-plda','plda-directory','audio-manifest','output'):
    parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);pins=PINS
for name,version in pins['versions'].items():
    assert importlib.metadata.version(name)==version, 'Package version: '+name
assert lfsha(Path(inspect.getfile(kaldi)))==pins['kaldi_lf_sha256'], 'Kaldi source'
model_paths=dict(segmentation=args.segmentation,encoder=args.embedding_directory/'embedding_encoder.onnx',projection=args.projection,plda=args.prepared_plda)
for name,path in model_paths.items():assert sha(path)==pins['models'][name], 'Model: '+name
for name,digest in pins['assets'].items():
    path=(args.embedding_directory if name.endswith('.npy') else args.plda_directory)/name
    assert sha(path)==digest, 'Asset: '+name
upstream=out/'upstream'
for name,digest in pins['source_hashes'].items():
    data=subprocess.check_output(['git','show',pins['source_revision']+':src/pyannote/audio/'+name],cwd=args.reference_source).replace(b'\r\n',b'\n')
    assert hashlib.sha256(data).hexdigest()==digest, 'Source: '+name
    path=upstream/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
configure(upstream)
torch.set_num_threads(1);torch.set_num_interop_threads(1)
options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0')
seg=ort.InferenceSession(str(model_paths['segmentation']),options,providers=['CPUExecutionProvider'])
enc=ort.InferenceSession(str(model_paths['encoder']),options,providers=['CPUExecutionProvider'])
p=upstream/'models/embedding/wespeaker/__init__.py'
f=next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='compute_fbank')
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[]);ns={'torch':torch};exec(compile(ast.fix_missing_locations(module),str(p),'exec'),ns)
settings=dict(num_mel_bins=80,frame_length=25.,frame_shift=10.,round_to_power_of_two=True,snip_edges=True,dither=0.,sample_frequency=16000,window_type='hamming',use_energy=False)
shim=types.SimpleNamespace(hparams=types.SimpleNamespace(fbank_centering_span=None),_fbank=functools.partial(kaldi.fbank,**settings))
p=upstream/'models/blocks/pooling.py';pn={};exec(compile(p.read_text(),str(p),'exec'),pn);pool=pn['StatsPool']()
w=torch.from_numpy(np.load(args.embedding_directory/'resnet_seg_1_weight.npy'));bias=torch.from_numpy(np.load(args.embedding_directory/'resnet_seg_1_bias.npy'))
source=upstream;vn={};exec(compile((source/'utils/vbx.py').read_text(),str(source/'utils/vbx.py'),'exec'),vn)
assets=args.plda_directory;tf,plda,phi=vn['vbx_setup'](assets/'xvec_transform.npz',assets/'plda.npz')
tree=ast.parse((source/'pipelines/clustering.py').read_text())
def method(cls,name):
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls);f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[])
    ns={'np':np,'linkage':linkage,'fcluster':fcluster,'linear_sum_assignment':linear_sum_assignment,'cdist':cdist,'rearrange':rearrange,'cluster_vbx':vn['cluster_vbx']}
    exec(compile(ast.fix_missing_locations(module),str(source/'pipelines/clustering.py'),'exec'),ns);return ns[name]
class Plda:
    def __init__(self):self.phi=phi
    def __call__(self,data):return plda(tf(data))
cluster=types.SimpleNamespace(plda=Plda(),threshold=.6,Fa=.07,Fb=.8,metric='cosine',constrained_assignment=True)
cluster.filter_embeddings=types.MethodType(method('BaseClustering','filter_embeddings'),cluster)
cluster.constrained_argmax=types.MethodType(method('BaseClustering','constrained_argmax'),cluster)
cluster_call=types.MethodType(method('VBxClustering','__call__'),cluster)
trials=[]
def model(waveform):
    try:
        features=ns['compute_fbank'](shim,waveform);encoded=enc.run(None,{'fbank_features':features.numpy()})[0]
        vector=torch.nn.functional.linear(pool(torch.from_numpy(encoded)),w,bias)
        trials.append(dict(samples=waveform.shape[-1],success=True,finite=bool(torch.isfinite(vector).all())))
        return vector
    except Exception as e:trials.append(dict(samples=waveform.shape[-1],success=False,error=str(e)));raise
env['cached_property']=functools.cached_property
extract('pipelines/speaker_verification.py','PyannoteAudioPretrainedSpeakerEmbedding',['min_num_samples'])
obj=env['PyannoteAudioPretrainedSpeakerEmbedding']();obj.sample_rate=16000;obj.device=torch.device('cpu');obj.model_=model
minimum=obj.min_num_samples;assert minimum==400
(out/'minimum-samples.json').write_text(json.dumps(dict(minimum=minimum,clean_threshold=2,trials=trials),indent=2))
class Stable:
    def __getattr__(self,name):return getattr(np,name)
    def argsort(self,a,axis=-1):return np.argsort(a,axis=axis,kind='stable')
files={}
def save(name,a):
    a=np.ascontiguousarray(a);assert np.isfinite(a).all(),name;p=out/(name+'.npy');np.save(p,a,allow_pickle=False)
    files[p.name]=dict(sha256=sha(p),shape=list(a.shape),dtype=str(a.dtype),bytes=p.stat().st_size);return p.name
speech_path=args.audio_manifest;speech=json.loads(speech_path.read_text());inputs=[]
if 'corpus' in pins:
    assert sha(speech_path)==pins['corpus']['manifest_sha256'], 'Corpus manifest'
    assert speech['revision']==pins['corpus']['revision'], 'Corpus source'
    assert [c['name'] for c in speech['cases']]==[c['name'] for c in pins['cases']], 'Corpus cases'
    for c,pin in zip(speech['cases'],pins['cases']):
        p=speech_path.parent/c['pcm'];assert sha(p)==c['pcm_sha256']==pin['pcm_sha256'], 'Corpus PCM'
        pcm=np.load(p,allow_pickle=False);assert pcm.dtype==np.float32 and pcm.shape==(pin['samples'],), 'Corpus layout'
        inputs.append((c['name'],pcm))
else:
    for name in ['english-16k','french-44k-stereo','jfk-48k-stereo']:
        c=next(c for c in speech['cases'] if c['name']==name);p=speech_path.parent/c['pcm'];assert sha(p)==c['pcm_sha256'];inputs.append((name,np.load(p).reshape(-1)))
    inputs.append(('two-recordings',np.concatenate([inputs[0][1],np.zeros(8000,np.float32),inputs[1][1]])))
    inputs.append(('silence',np.zeros(16000,np.float32)))
cases=[]
for name,pcm in inputs:
    case=dict(name=name,pcm=save(name+'-pcm',pcm),seconds=len(pcm)/16000,windows=[],status='Completed');waveforms=windows(pcm);activity=[]
    for i,wave in enumerate(waveforms):
        scores=seg.run(None,{'waveform':wave[None]})[0];assert np.array_equal(scores,seg.run(None,{'waveform':wave[None]})[0])
        a=powerset(scores)[0];activity.append(a);case['windows'].append(dict(scores=save(f'{name}-{i}-scores',scores),activity=save(f'{name}-{i}-activity',a)))
    activity=np.stack(activity)
    count=env['SpeakerDiarizationMixin'].speaker_count(env['SlidingWindowFeature'](activity.copy(),env['SlidingWindow'](duration=10,step=1)),env['SlidingWindow'](duration=991/16000,step=270/16000),warm_up=(0.,0.)).data
    case['count']=save(name+'-count',count)
    if count.max()==0:
        case['status']='NoSpeech';case['intervals']=case['exclusive_intervals']=[];case['centroids']=save(name+'-centroids',np.empty((0,256),np.float64));cases.append(case);print(name,'NoSpeech',flush=True);continue
    used,embedding_windows=masks(activity,pcm);assert np.array_equal(embedding_windows,np.repeat(waveforms,3,axis=0));vectors=[]
    for i,wave in enumerate(waveforms):
        fbank=ns['compute_fbank'](shim,torch.from_numpy(wave[None])).numpy();encoded=enc.run(None,{'fbank_features':fbank})[0]
        assert np.array_equal(encoded,enc.run(None,{'fbank_features':fbank})[0])
        case['windows'][i].update(features=save(f'{name}-{i}-features',fbank),encoded=save(f'{name}-{i}-encoded',encoded),masks=save(f'{name}-{i}-masks',used[i*3:i*3+3]))
        pooled=[];v=[]
        for mask in used[i*3:i*3+3]:
            p=pool(torch.from_numpy(encoded),torch.from_numpy(mask[None]));pooled.append(p.numpy()[0]);v.append(torch.nn.functional.linear(p,w,bias).numpy()[0])
        v=np.stack(v);vectors.append(v);case['windows'][i].update(pooled=save(f'{name}-{i}-pooled',np.stack(pooled)),vectors=save(f'{name}-{i}-vectors',v))
    vectors=np.stack(vectors);hard,soft,centers=cluster_call(embeddings=vectors,segmentations=types.SimpleNamespace(data=activity),min_clusters=1,max_clusters=np.inf)
    hard[~activity.any(axis=1)]=-2;original_hard=hard.copy();order={}
    for v in hard.reshape(-1):
        if v>=0 and int(v) not in order:order[int(v)]=len(order)
    for v in range(len(centers)):
        if v not in order:order[v]=len(order)
    canonical=np.asarray([order.get(int(v),-2) for v in hard.reshape(-1)]).reshape(hard.shape)
    ordered=np.empty_like(centers)
    for old,new in order.items():ordered[new]=centers[old]
    case['labels']=save(name+'-labels',canonical);case['original_labels']=save(name+'-original-labels',original_hard);case['centroids']=save(name+'-centroids',ordered)
    _,raw=timeline(activity,canonical)
    env['np']=Stable()
    try:_,policy=timeline(activity,canonical)
    finally:env['np']=np
    for kind in ['ordinary','exclusive']:
        case[kind+'_native_frames']=save(name+'-'+kind+'-native-frames',raw[kind]['binary']);case[kind+'_frames']=save(name+'-'+kind+'-frames',policy[kind]['binary'])
        case[kind+'_native_intervals']=raw[kind]['intervals']
    def clip(intervals):return [[max(0,s),min(case['seconds'],e),k] for s,e,k in intervals if min(case['seconds'],e)>max(0,s)]
    a=clip(policy['ordinary']['intervals']);b=clip(policy['exclusive']['intervals']);visible=sorted(set(v[2] for v in a));mapping={old:new for new,old in enumerate(visible)}
    case['intervals']=[[s,e,mapping[k]] for s,e,k in a];case['exclusive_intervals']=[[s,e,mapping[k]] for s,e,k in b]
    case['speakers']=[dict(speaker=mapping[k],centroid=ordered[k].tolist() if k<len(ordered) else [0.]*256,has_embedding=k<len(ordered)) for k in visible]
    if not visible:case['status']='NoSpeech'
    cases.append(case);print(name,len(waveforms),'windows',len(ordered),'clusters',len(a),'intervals',flush=True)
record=dict(scope='Connected automatic Community-1 with deterministic vote ties and PCM-bound intervals; original native timelines retained',
    pins=pins,recipes=recipes(),models={k:sha(v) for k,v in model_paths.items()},cases=cases,files=files,
    numpy=np.__version__,torch=torch.__version__,onnxruntime=ort.__version__,speech_manifest_sha256=sha(speech_path))
assert {k:{a:v[a] for a in ('shape','dtype')} for k,v in files.items()} == pins['layout'], 'Reference layout changed'
with (out/'manifest.json').open('x') as f:json.dump(record,f,indent=2)
print('Complete',len(cases),'cases',len(files),'files')
