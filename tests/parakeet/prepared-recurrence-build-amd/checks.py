"""Require the declared lifecycle/dispatch scope and exact unrelated compiled code."""
import json

# Fully named existing methods whose source is changed by the frozen generator.
REQUIRED = {
    ('Lokad.Onnx.CPUExecutionProvider','Lstm'),
    ('Lokad.Onnx.ComputationalGraph','InvalidatePreparation'),
    ('Lokad.Onnx.ComputationalGraph','RunCoreInner'),
    ('Lokad.Onnx.ComputationalGraph','RefreshLifetimeAnalysis'),
    ('Lokad.Onnx.GraphExecution','.ctor'),
    ('Lokad.Onnx.GraphPacking','PackMatMulWeights'),
    ('Lokad.Onnx.TensorExecutionOptions','GetHashCode'),
}
GENERATED = {('Lokad.Onnx.TensorExecutionOptions','.ctor'),('Lokad.Onnx.TensorExecutionOptions','Equals')}
NEW_TYPES = ['Lokad.Onnx.GraphLstmPacking','Lokad.Onnx.PackedLstmWeight']


def identity(key):return tuple(key.split('::',2)[:2])


def inventory(value,measured,built,prior):
    assert value['inventory_complete'] and len(value['observations'])==2 and prior['passed']
    result=None
    for row,(name,count) in zip(value['observations'],[('Lokad.Onnx.dll',3189),('Lokad.Onnx.Data.dll',697)],strict=True):
        assert row['assembly']==name and row['methods']==len(row['normalized_methods'])==count
        assert row['before_sha256']==measured[name]['sha256'] and row['after_sha256']==built[name]['sha256']
        assert row['public_surface_equal'] and row['compiler_rename'] is None and not row['removed']
        before=row['normalized_methods'];after=before|row['candidate_methods']
        assert before==prior['methods'][name] and row['method_flags_before']==prior['flags'][name]
        assert set(row['method_flags_before'])==set(before) and set(row['method_flags_after'])==set(after)
        assert all(row['method_flags_after'][k]==flag for k,flag in row['method_flags_before'].items())
        assert set(row['candidate_methods'])==set(row['differences'])|set(row['added'])
        assert row['unchanged_methods']==count-len(row['differences'])
        assert all(before[k]!=after[k] for k in row['differences'])
        if name=='Lokad.Onnx.Data.dll':
            assert row['unchanged_methods']==697 and not row['differences'] and not row['added'];continue
        changes={identity(k) for k in row['differences']}
        assert REQUIRED<=changes
        assert changes<=REQUIRED|GENERATED|{('Lokad.Onnx.ComputationalGraph','.ctor')}
        constructors=[k for k in row['differences'] if identity(k)==('Lokad.Onnx.ComputationalGraph','.ctor')]
        assert constructors==['Lokad.Onnx.ComputationalGraph::.ctor::Void .ctor(Int64)']
        # A record's typed equality and initialization can change with the new
        # internal field. Its object equality wrapper must remain unchanged.
        equal=[k for k in row['differences'] if identity(k)==('Lokad.Onnx.TensorExecutionOptions','Equals')]
        assert equal==['Lokad.Onnx.TensorExecutionOptions::Equals::Boolean Equals(Lokad.Onnx.TensorExecutionOptions)']
        assert len(row['differences'])==len(changes)
        for key in row['added']:
            declaring,method,_=key.split('::',2)
            assert any(declaring==t or declaring.startswith(t+'+') for t in NEW_TYPES) or (
                declaring=='Lokad.Onnx.TensorExecutionOptions' and method in ['get_PackedLstmWeights','set_PackedLstmWeights'])
            assert row['method_flags_after'][key]==0
            assert after[key]!='NO-BODY'
        added={identity(k) for k in row['added']}
        assert {('Lokad.Onnx.TensorExecutionOptions','get_PackedLstmWeights'),('Lokad.Onnx.TensorExecutionOptions','set_PackedLstmWeights'),
                ('Lokad.Onnx.GraphLstmPacking','Resolve'),('Lokad.Onnx.GraphLstmPacking','PruneAndBytes'),('Lokad.Onnx.GraphLstmPacking','PackWeights')}<=added
        kernel=[k for k in before if identity(k)==('Lokad.Onnx.CPUExecutionProvider','LstmProjectOrdered')]
        assert len(kernel)==1 and before[kernel[0]]==after[kernel[0]]
        assert not any('LstmProjectionPanels' in key or 'LstmProjectOrderedRows' in key for key in row['differences']+row['added'])
        lstm=next(k for k in row['differences'] if identity(k)==('Lokad.Onnx.CPUExecutionProvider','Lstm'))
        calls=[i['operand'] for i in json.loads(after[lstm])['instructions'] if i['opcode']=='call']
        assert sum('GraphLstmPacking::' in target and ' Resolve(' in target for target in calls)==2
        assert sum('LstmProjectOrdered(' in target for target in calls)==2
        result=dict(passed=True,original_core_methods=count,candidate_core_methods=len(after),data_methods=697,
            existing_method_changes=row['differences'],added_methods=row['added'],unchanged_core_methods=row['unchanged_methods'],
            existing_flags_exact=True,new_flags={k:row['method_flags_after'][k] for k in row['added']},
            public_surface_equal=True,ordered_projection_exact=True,existing_panels_exact=True,
            source_prepared=prior['source_prepared'],numerically_qualified=False)
    assert result is not None
    return result
