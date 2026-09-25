"""Account for one deleted Tensor method without hiding executable changes."""
import copy
import json
import re

ORDINALS = {497, 498, 499, 500, 501, 521, 522, 525, 529, 530, 531, 544, 545}
PATTERN = re.compile(r'(<>c__DisplayClass|<>9__|>b__|>g__[^|<>]+\|)(\d+)(?=_)')
MEMBER_OPS = {'call','callvirt','newobj','ldftn','ldvirtftn','ldfld','ldflda','stfld','ldsfld','ldsflda',
              'stsfld','ldtoken','castclass','isinst','initobj','box','unbox','unbox.any','sizeof','constrained.'}
TENSOR = 'Lokad.Onnx.Tensor`1[T]'
DELETED = TENSOR+'::OwnedRemainderSource::Lokad.Onnx.DenseTensor`1[System.Single] OwnedRemainderSource(Lokad.Onnx.OwnedPackedTensor, Int32, Lokad.Onnx.TensorExecutionOptions)'
RUN_OLD = TENSOR+'::RunOwnedPackedRows::Void RunOwnedPackedRows(Int32, Int32, Int32, Single*, Single*, Single*, Single*)'
RUN_NEW = TENSOR+'::RunOwnedPackedRows::Void RunOwnedPackedRows(Int32, Int32, Int32, Single*, Single*, Single*)'
HELPER = 'Lokad.Onnx.PackedFinalRowKernel::Multiply::Void Multiply(Int32, Int32, Single*, Single*, Single*)'


def renamed(value):
    if not value.startswith('Lokad.Onnx.Tensor`1'): return value
    def replace(match):
        number = int(match[2])
        return match[1]+str(number-1 if number in ORDINALS else number)
    return PATTERN.sub(replace,value)


def body_after_rename(value):
    if value=='NO-BODY': return value
    body=json.loads(value)
    for local in body['locals']: local['type']=renamed(local['type'])
    for clause in body['exceptions']:
        if clause['caught'] is not None: clause['caught']=renamed(clause['caught'])
    for instruction in body['instructions']:
        if instruction['opcode'] in MEMBER_OPS: instruction['operand']=renamed(instruction['operand'])
    return body


def reconcile(row):
    if row['assembly']!='Lokad.Onnx.dll':return dict(row,compiler_renames={},signature_changes={})
    old=row['normalized_methods']
    assert DELETED in old and RUN_OLD in old and RUN_NEW not in old
    candidate={key:value for key,value in old.items() if key not in row['removed']}
    candidate.update(row['candidate_methods'])
    assert set(candidate)==set(row['method_flags_after'])
    assert DELETED not in candidate and RUN_OLD not in candidate and RUN_NEW in candidate
    mapping={key:(RUN_NEW if key==RUN_OLD else renamed(key)) for key in old if key!=DELETED}
    assert len(set(mapping.values()))==len(mapping),'Mapping must be injective'
    assert set(mapping.values())<=set(candidate),'Every other original method must survive'
    differences=[]
    for key,target in mapping.items():
        actual=candidate[target]
        if actual!='NO-BODY':actual=json.loads(actual)
        if body_after_rename(old[key])!=actual:differences.append(key)
        assert row['method_flags_before'][key]==row['method_flags_after'][target],key
    added=sorted(set(candidate)-set(mapping.values()))
    result=copy.deepcopy(row)
    result.update(removed=[DELETED],added=added,differences=differences,
        unchanged_methods=len(old)-1-len(differences),
        compiler_renames={key:target for key,target in mapping.items() if key!=target and key!=RUN_OLD},
        signature_changes={RUN_OLD:RUN_NEW})
    result['method_flags_after']={key:row['method_flags_after'][target] for key,target in mapping.items()}
    result['method_flags_after'].update({key:row['method_flags_after'][key] for key in added})
    result['candidate_methods']={key:candidate[mapping[key]] for key in differences}
    result['candidate_methods'].update({key:candidate[key] for key in added})
    return result
