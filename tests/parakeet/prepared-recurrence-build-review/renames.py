"""Exact compiler-name reconciliation after adding the graph's internal map field."""
import copy
import json
import re

# These replacements affect identifiers only, inside ComputationalGraph bodies.
# In particular, the two ResolveInputs overloads both move one ordinal; one
# new identifier collides with the other overload's former identifier.
FORWARD = {
    '<ResolveInputs>b__183_0':'<ResolveInputs>b__184_0',
    '<ResolveInputs>b__184_0':'<ResolveInputs>b__185_0',
    '<ResolveInputs>b__183_1':'<ResolveInputs>b__184_1',
    '<ResolveNodeExecuteInputs>b__185_0':'<ResolveNodeExecuteInputs>b__186_0',
    '<ResolveNodeExecuteInputs>b__186_0':'<ResolveNodeExecuteInputs>b__187_0',
    '<RunCoreInner>b__190_0':'<RunCoreInner>b__191_0',
    '<RunCoreInner>b__190_2':'<RunCoreInner>b__191_2',
    '<RunNodeCoreInner>b__200_1':'<RunNodeCoreInner>b__201_1',
    '<>c__DisplayClass190_0':'<>c__DisplayClass191_0',
    '<>c__DisplayClass200_0':'<>c__DisplayClass201_0',
    '<>c__DisplayClass211_0':'<>c__DisplayClass212_0',
    '<>c__DisplayClass211_1':'<>c__DisplayClass212_1',
    '<ComputeStructureFingerprint>g__MixUlong|211_0':'<ComputeStructureFingerprint>g__MixUlong|212_0',
    '<ComputeStructureFingerprint>g__MixInt|211_1':'<ComputeStructureFingerprint>g__MixInt|212_1',
    '<ComputeStructureFingerprint>g__MixString|211_2':'<ComputeStructureFingerprint>g__MixString|212_2',
    '<EnumerateAttributeTensors>d__233':'<EnumerateAttributeTensors>d__234',
    '<>9__183_1':'<>9__184_1',
    '<>9__185_0':'<>9__186_0',
    '<>9__186_0':'<>9__187_0',
    '<>9__190_0':'<>9__191_0',
    '<>9__190_2':'<>9__191_2',
    '<>9__200_1':'<>9__201_1',
}


def normalize(value):
    inverse={new:old for old,new in FORWARD.items()};assert len(inverse)==len(FORWARD)
    pattern=re.compile('|'.join(re.escape(k) for k in sorted(inverse,key=len,reverse=True)))
    def change(value):
        if isinstance(value,str):return pattern.sub(lambda m:inverse[m.group()],value)
        if isinstance(value,list):return [change(v) for v in value]
        if isinstance(value,dict):return {k:change(v) for k,v in value.items()}
        return value
    result=copy.deepcopy(value);row=result['observations'][0]
    assert row['assembly']=='Lokad.Onnx.dll' and row['compiler_rename'] is None
    assert len(row['removed'])==27 and len(row['added'])==88 and len(row['differences'])==17
    before=row['normalized_methods'];actual={k:v for k,v in before.items() if k not in row['removed']}|row['candidate_methods']
    methods={};flags={};renamed=[];name_only=[]
    for key,body in actual.items():
        graph=key.startswith('Lokad.Onnx.ComputationalGraph::') or key.startswith('Lokad.Onnx.ComputationalGraph+')
        adjusted=change(key) if graph else key
        changed_body=json.dumps(change(json.loads(body)),separators=(',',':')) if graph and body!='NO-BODY' else body
        if adjusted in before and body!='NO-BODY' and json.loads(changed_body)==json.loads(before[adjusted]):
            changed_body=before[adjusted]
            if key!=adjusted or body!=changed_body:name_only.append(dict(before=adjusted,after=key))
        assert adjusted not in methods
        methods[adjusted]=changed_body;flags[adjusted]=row['method_flags_after'][key]
        if adjusted!=key:
            assert adjusted in before and changed_body==before[adjusted],(adjusted,key)
            assert flags[adjusted]==row['method_flags_before'][adjusted]
            renamed.append(dict(before=adjusted,after=key))
    assert set(before)<=set(methods)
    differences=[k for k,v in before.items() if methods[k]!=v]
    added=[k for k in methods if k not in before]
    row.update(differences=differences,added=added,removed=[],candidate_methods={k:methods[k] for k in differences+added},
               method_flags_after=flags,unchanged_methods=len(before)-len(differences))
    return result,dict(renamed_methods=renamed,name_only_methods=name_only,identifier_substitutions=FORWARD)
