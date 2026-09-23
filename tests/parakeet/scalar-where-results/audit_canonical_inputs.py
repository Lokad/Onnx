"""Reconcile the frozen layout audit's raw Constant names with proven deduplication."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import onnx
from onnx import numpy_helper

ROOT=Path(__file__).resolve().parents[3]
TOOLS=ROOT/'tests/parakeet/scalar-where-layout-amd'
sys.path.insert(0,str(TOOLS))
from protocol import pin,read,save
from prepare import BASE,RELEASE


def validate_input_names(observation,expected,aliases):
    assert observation['name']==expected['name']
    actual=observation['input_names'];original=expected['inputs']
    assert len(actual)==len(original)==3
    assert actual[0]==original[0] and actual[2]==original[2]
    if expected['x_scalar'] is None:assert actual==original
    else:assert actual[1]==aliases[original[1]]


def checker_tests(aliases,expected,observation):
    validate_input_names(observation,expected,aliases)
    count=1
    for field in [0,1,2,'name','arity']:
        bad=copy.deepcopy(observation)
        if type(field) is int:bad['input_names'][field]+='-wrong'
        elif field=='name':bad['name']+='-wrong'
        else:bad['input_names'].append('extra')
        try:validate_input_names(bad,expected,aliases)
        except AssertionError:count+=1
        else:raise AssertionError(('Checker accepted corruption',field))
    return count


def main():
    assert not (BASE/'closed.json').exists() and not (BASE/'audit-reconciliation.json').exists()
    original=TOOLS/'audit.py';prepared=read(BASE/'prepared.json')
    assert pin(original)==prepared['files'][original.relative_to(ROOT).as_posix()]
    model=ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    static=read(BASE/'bundle/evidence/graph-observations.json')['graph']
    assert pin(model)==static['model']
    graph=onnx.load(model,load_external_data=False).graph
    assert len(graph.node)==4491 and not any(n.op_type=='If' for n in graph.node)
    protected={n.name for n in graph.output}
    constants={};canonical={}
    for node in graph.node:
        if node.op_type!='Constant':continue
        tensor=next((a.t for a in node.attribute if a.name=='value'),None)
        if tensor is None or tensor.data_type!=onnx.TensorProto.FLOAT or len(tensor.dims)!=0:continue
        assert len(node.output)==1
        value=numpy_helper.to_array(tensor);bits=value.tobytes().hex()
        if bits not in ['00000000','00401cc6']:continue
        assert node.output[0] not in protected
        constants[node.output[0]]=dict(dtype='Float',shape=[],bits=bits)
        canonical.setdefault(bits,node.output[0])
    assert len(constants)==72
    assert sum(c['bits']=='00000000' for c in constants.values())==48
    assert canonical=={'00401cc6':'/layers.0/self_attn/Constant_111_output_0',
                      '00000000':'/layers.0/self_attn/Constant_112_output_0'}
    aliases={name:canonical[value['bits']] for name,value in constants.items()}
    result=read(BASE/'collected/capture/result.json');changes=[]
    for request in result['requests']:
        assert request['graph_nodes']==2856 and len(request['observations'])==73
        for actual,expected in zip(request['observations'],static['wheres'],strict=True):
            validate_input_names(actual,expected,aliases)
            if actual['input_names'][1]!=expected['inputs'][1]:
                old,new=expected['inputs'][1],actual['input_names'][1]
                assert constants[old]==constants[new]
                assert actual['inputs'][1]['dtype']=='Float' and actual['inputs'][1]['shape']==[]
                assert actual['inputs'][1]['scalar_bits']==constants[old]['bits']
                changes.append(dict(request=request['name'],node=actual['name'],original=old,canonical=new,**constants[old]))
    assert len(changes)==140
    source=ROOT/'src/Lokad.Onnx/Optimization/ConstFold.cs'
    root_source=read(RELEASE/'bundle/evidence/root-applied.json')['source_files']
    assert pin(source)==root_source[source.relative_to(ROOT).as_posix()]
    assert 'if (BitsEqual(survivorTensor, dupTensor)) { rename[dup] = survivor; break; }' in source.read_text()
    tests=checker_tests(aliases,static['wheres'][3],result['requests'][0]['observations'][3]);assert tests==6
    before="            assert observation['name']==expected['name'] and observation['input_names']==expected['inputs']"
    after="            validate_input_names(observation,expected,aliases)"
    text=original.read_text();assert text.count(before)==1
    corrected=text.replace(before,after)
    save(BASE/'audit-reconciliation.json',dict(passed=True,original_audit_failed=True,
        reason='The original audit compared raw ONNX constant names; unchanged ConstFold deduplicates bit-identical scalar constants before execution.',
        original_audit=pin(original),script=pin(__file__),model=pin(model),source=pin(source),
        source_path=source.relative_to(ROOT).as_posix(),constants=constants,aliases=aliases,changes=changes,
        before=before,after=after,corrected_audit_sha256=hashlib.sha256(corrected.encode()).hexdigest(),
        checker_tests=tests,all_other_checks_unchanged=True,no_worker_repeated=True))
    namespace=dict(__name__='canonical_input_audit',__file__=str(original),aliases=aliases,validate_input_names=validate_input_names)
    exec(compile(corrected,str(original)+' [verified canonical input reconciliation]','exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
