from pathlib import Path
import ast,functools,hashlib,itertools,math,sys,types,warnings
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pyannote.core import Annotation,Segment,SlidingWindow,SlidingWindowFeature

source = Path()
env=dict(np=np,torch=torch,F=F,math=math,warnings=warnings,rearrange=rearrange,itertools=itertools,
    Annotation=Annotation,Segment=Segment,SlidingWindow=SlidingWindow,SlidingWindowFeature=SlidingWindowFeature)
def extract(path,class_name,names,new_name=None,bases=()):
    p=source/path;tree=ast.parse(p.read_text(encoding='utf-8'))
    if class_name:
        c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
        methods=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in methods}==set(names)
        body=[ast.ClassDef(name=new_name or class_name,bases=[ast.Name(id=b,ctx=ast.Load()) for b in bases],keywords=[],body=methods,decorator_list=[])]
    else:body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+body,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(p),'exec'),env)
def configure(directory):
    """Load methods only from the caller's hash-verified upstream snapshot."""
    global source, Inference, Pipeline, Powerset, frames, chunks
    source = directory
    extract('core/inference.py','Inference',['aggregate','trim','slide'])
    extract('utils/signal.py','Binarize',['__init__','__call__'])
    extract('pipelines/utils/diarization.py','SpeakerDiarizationMixin',['speaker_count','to_diarization','to_annotation'])
    extract('pipelines/speaker_diarization.py',None,['batchify'])
    extract('pipelines/speaker_diarization.py','SpeakerDiarization',['reconstruct','get_embeddings'],bases=('SpeakerDiarizationMixin',))
    extract('core/io.py','Audio',['crop'])
    extract('utils/powerset.py','Powerset',['build_mapping','to_multilabel'])
    env['Inference'].duration=10.;env['Inference'].step=1.
    env['Resolution']=types.SimpleNamespace(CHUNK='chunk')
    env['map_with_specifications']=lambda specs,func,*values:func(*values,specifications=specs)
    Inference=env['Inference'];Pipeline=env['SpeakerDiarization'];Powerset=env['Powerset']
    env['combinations']=itertools.combinations
    frames=SlidingWindow(start=0.,duration=991/16000,step=270/16000)
    chunks=SlidingWindow(start=0.,duration=10.,step=1.)

def feature(a):return SlidingWindowFeature(a.copy(),chunks)
def powerset(logits):
    obj=Powerset();obj.num_classes=3;obj.max_set_size=2;obj.num_powerset_classes=7;obj.mapping=obj.build_mapping()
    return obj.to_multilabel(torch.from_numpy(logits)).numpy()
def intervals(binary):
    annotation=Pipeline.to_annotation(binary,min_duration_on=0.,min_duration_off=0.)
    return [[s.start,s.end,int(label)] for s,_,label in annotation.itertracks(yield_label=True)]
def timeline(activity,labels):
    count=Pipeline.speaker_count(feature(activity),frames,warm_up=(0.,0.))
    results={}
    for name,cap in [('ordinary',127),('exclusive',1)]:
        c=SlidingWindowFeature(np.minimum(count.data,cap).astype(np.int8),count.sliding_window)
        binary=Pipeline().reconstruct(feature(activity),labels,c)
        results[name]=dict(binary=binary.data,intervals=intervals(binary),start=binary.sliding_window.start,step=binary.sliding_window.step,duration=binary.sliding_window.duration)
    return count.data,results
def windows(samples):
    obj=Inference();obj.model=types.SimpleNamespace(audio=types.SimpleNamespace(get_num_samples=lambda seconds:round(seconds*16000)),
        specifications=types.SimpleNamespace(resolution='frame'),receptive_field=frames)
    obj.batch_size=3;obj.skip_aggregation=True;obj.pre_aggregation_hook=None;captured=[]
    def infer(batch):
        captured.extend(v.numpy().copy() for v in batch)
        return np.zeros((len(batch),589,3),np.float32)
    obj.infer=infer;obj.slide(torch.from_numpy(samples.reshape(1,-1)),16000,None)
    return np.stack(captured)
def masks(activity,samples):
    obj=Pipeline();obj.training=False;obj.embedding_batch_size=4
    audio=env['Audio']();audio.validate_file=lambda f:f;audio.get_num_samples=lambda seconds,rate:round(seconds*rate);audio.downmix_and_resample=lambda data,rate,channel:(data,rate)
    obj._audio=audio;captured=[];waveforms=[]
    class Embedding:
        min_num_samples=400;sample_rate=16000
        def __call__(self,batch,masks):
            captured.extend(v.numpy().copy() for v in masks);waveforms.extend(v.numpy().copy() for v in batch)
            return np.zeros((len(batch),256),np.float32)
    obj._embedding=Embedding();obj.get_embeddings(dict(waveform=torch.from_numpy(samples.reshape(1,-1)),sample_rate=16000),feature(activity),exclude_overlap=True)
    return np.stack(captured),np.stack(waveforms)
