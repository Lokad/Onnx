from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,subprocess,types
here=Path(__file__).resolve().parent;root=here.parents[2]
parser=argparse.ArgumentParser(description='Pinned complete Community-1 clustering references, without downloads.')
parser.add_argument('--reference-source',type=Path,required=True)
parser.add_argument('--model-directory',type=Path,required=True)
parser.add_argument('--speaker-reference',type=Path,required=True)
parser.add_argument('--managed-speaker-result',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from einops import rearrange
assert np.__version__=='2.2.4' and scipy.__version__=='1.16.3'
out=args.output;out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=out
paths={'vbx.py':'src/pyannote/audio/utils/vbx.py','clustering.py':'src/pyannote/audio/pipelines/clustering.py'}
sources={}
for name,path in paths.items():
    data=subprocess.check_output(['git','-C',str(args.reference_source),'show','a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a:'+path])
    expected={'vbx.py':'812c8c4ba276ba0521689693f84690f6fc9af3d67dec232aac45c1fc2d41906f','clustering.py':'6031fb7c21277a7e9901ef2cdaed7d5cd69f7ef45508dc4b45e82ce0da3c8fba'}
    assert hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==expected[name]
    (out/name).write_bytes(data);sources[name]=hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()
ns={};exec(compile((source/'vbx.py').read_text(encoding='utf-8'),str(source/'vbx.py'),'exec'),ns)
assets=args.model_directory
assert sha(assets/'plda.npz')=='9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255'
assert sha(assets/'xvec_transform.npz')=='325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f'
tf,plda,phi=ns['vbx_setup'](assets/'xvec_transform.npz',assets/'plda.npz')
closed=lambda fn:dict(zip(fn.__code__.co_freevars,[c.cell_contents for c in fn.__closure__]))
x=closed(tf);p=closed(plda)
prepared=dict(schema=1,input_dimensions=256,output_dimensions=128,mean1=x['mean1'].tolist(),mean2=x['mean2'].tolist(),lda=x['lda'].tolist(),mean=p['plda_mu'].tolist(),transform=p['plda_tr'].tolist(),phi=phi.tolist(),sources=sources,
    assets={n:sha(assets/n) for n in ['plda.npz','xvec_transform.npz']},numpy=np.__version__,scipy=scipy.__version__)
prepare_path=root/'eng/prepare-community1-plda.py'
spec=importlib.util.spec_from_file_location('prepare_plda',prepare_path);prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
preparation=prepare.prepare(assets,out/'prepared.json')
actual_prepared=json.loads((out/'prepared.json').read_text(encoding='utf-8'))
for key in ('mean1','mean2','lda','mean','transform','phi'):assert np.array_equal(np.asarray(prepared[key]),np.asarray(actual_prepared[key]))
# Execute original methods with their original SciPy/NumPy functions. No reimplementation is the native oracle.
tree=ast.parse((source/'clustering.py').read_text(encoding='utf-8'))
def method(class_name,method_name):
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method_name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[])
    env={'np':np,'linkage':linkage,'fcluster':fcluster,'linear_sum_assignment':linear_sum_assignment,'cdist':cdist,'rearrange':rearrange,'cluster_vbx':ns['cluster_vbx']}
    exec(compile(ast.fix_missing_locations(module),str(source/'clustering.py'),'exec'),env);return env[method_name]
class Plda:
    def __init__(self):self.phi=phi
    def __call__(self,data):return plda(tf(data))
obj=types.SimpleNamespace(plda=Plda(),threshold=.6,Fa=.07,Fb=.8,metric='cosine',constrained_assignment=True)
obj.filter_embeddings=types.MethodType(method('BaseClustering','filter_embeddings'),obj)
obj.constrained_argmax=types.MethodType(method('BaseClustering','constrained_argmax'),obj)
call=types.MethodType(method('VBxClustering','__call__'),obj)
old=args.speaker_reference;m=json.loads((old/'manifest.json').read_text());native=[]
for name in ['english-16k-none','french-44k-stereo-none','jfk-48k-stereo-none','noise-16000-none','tones-none']:
    c=next(c for c in m['cases'] if c['name']==name);p=old/c['vector'];assert sha(p)==m['files'][c['vector']]['sha256'];native.append(np.load(p).reshape(-1))
native=np.stack(native)
local_report=args.managed_speaker_result;lm=json.loads(local_report.read_text());managed=[]
for name in ['english-16k-none','french-44k-stereo-none','jfk-48k-stereo-none','noise-16000-none','tones-none']:
    c=next(c for c in lm['reports'] if c['name']==name and c['repeat']==0);p=Path(str(local_report)+'.arrays')/c['file'];assert sha(p)==c['sha256'];managed.append(np.fromfile(p,np.float32))
managed=np.stack(managed)
rng=np.random.default_rng(20260919);cases=[]
def array(x):
    a=np.asarray(x);return dict(shape=list(a.shape),dtype=str(a.dtype),values=a.reshape(-1).tolist())
def add(name,embeddings,segmentation):
    embeddings=np.ascontiguousarray(embeddings,np.float32);segmentation=np.ascontiguousarray(segmentation,np.float32)
    e0=embeddings.copy();s0=segmentation.copy();segmentations=types.SimpleNamespace(data=segmentation)
    train,ci,si=obj.filter_embeddings(embeddings,segmentations)
    if len(train)>=2:
        norm=train/np.linalg.norm(train,axis=1,keepdims=True);z=linkage(norm,method='centroid',metric='euclidean');labels=fcluster(z,.6,criterion='distance')-1;_,labels=np.unique(labels,return_inverse=True)
        transformed=obj.plda(train);qinit=np.zeros((len(labels),max(labels)+1));qinit[np.arange(len(labels)),labels]=1
        qinit=ns['softmax'](qinit*7,axis=1)
        q,priors,trace,alpha,inv=ns['VBx'](transformed,phi,Fa=.07,Fb=.8,pi=qinit.shape[1],gamma=qinit,maxIters=20,return_model=True)
        extra=dict(normalized=array(norm),hierarchy=array(z),labels=array(labels),transformed=array(transformed),responsibilities=array(q),priors=array(priors),objective=array(trace),alpha=array(alpha),inverse_precision=array(inv))
    else:extra={}
    hard,soft,centroids=call(embeddings,segmentations,min_clusters=1,max_clusters=1000)
    again=call(embeddings,segmentations,min_clusters=1,max_clusters=1000)
    assert all(np.array_equal(a,b,equal_nan=True) for a,b in zip((hard,soft,centroids),again))
    assert np.array_equal(embeddings,e0,equal_nan=True) and np.array_equal(segmentation,s0)
    # Inputs here are finite; explicit empty/NaN policy is covered in separate ordinary tests.
    assert all(np.isfinite(a).all() for a in (hard,soft,centroids))
    cases.append(dict(name=name,embeddings=array(embeddings),activity=array(segmentation),training_indices=array(ci*embeddings.shape[1]+si),hard=array(hard),soft=array(soft),centroids=array(centroids),**extra))
    print(name,'training',len(train),'clusters',len(centroids),'iterations',len(extra.get('objective',{}).get('values',[])),flush=True)
def activity(chunks,speakers,frames=30):
    a=np.zeros((chunks,frames,speakers),np.float32)
    for s in range(speakers):a[:,s*frames//speakers:(s+1)*frames//speakers,s]=1
    return a
perturbation=rng.normal(0,.001,(8,3,256)).astype(np.float32)
for prefix,voices in [('native',native),('managed',managed)]:
    add(prefix+'-recorded',np.stack([voices[[0,1,2]],voices[[2,0,1]],voices[[1,2,0]]]),activity(3,3))
    add(prefix+'-diverse',np.stack([voices[[0,3,4]],voices[[3,4,0]],voices[[4,0,3]]]),activity(3,3))
    clustered=voices[[0,1,2]][None]+perturbation
    add(prefix+'-perturbed',clustered,activity(8,3))
    a=activity(4,3);a[0,:,2]=0;a[1,:,1]=0;a[2,:8,1]=1;a[3,:,0]=1 # includes inactive and overlap-filtered rows.
    add(prefix+'-filtered',np.stack([voices[[0,1,2]] for _ in range(4)]),a)
add('one-training',native[:1].reshape(1,1,256),activity(1,1))
add('duplicate',np.tile(native[0],(4,3,1)),activity(4,3))
for count in [2,4,16,64,128]:
    values=rng.normal(size=(count,1,256)).astype(np.float32);add('random-'+str(count),values,activity(count,1))
# Hierarchy-only adversarial geometry, independently generated to contain centroid inversions.
hierarchy=[]
for name,points in [('duplicate',np.array([[0,0],[0,0],[1,0],[1,0]],np.float32)),('inversion',np.array([[0,0],[1,0],[.5,.86]],np.float32)),('square',np.array([[0,0],[0,1],[1,0],[1,1]],np.float32))]:
    z=linkage(points,method='centroid',metric='euclidean')
    for threshold in [0.,.6,.9,1.,1.1]:
        labels=fcluster(z,threshold,criterion='distance')-1
        hierarchy.append(dict(name=name+'-'+str(threshold),points=array(points),threshold=threshold,hierarchy=array(z),labels=array(labels)))
record=dict(scope='Community-1 automatic speaker clustering, not audio timelines',source_revision='a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a',source_hashes=sources,prepared_sha256=sha(out/'prepared.json'),numpy=np.__version__,scipy=scipy.__version__,generator_lf_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),prepare_lf_sha256=hashlib.sha256(prepare_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),preparation=preparation,speaker_manifest_sha256=sha(old/'manifest.json'),managed_speaker_result_sha256=sha(local_report),cases=cases,hierarchy=hierarchy)
(out/'reference.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print('Complete',len(cases),'pipeline cases',len(hierarchy),'hierarchy cases',flush=True)
