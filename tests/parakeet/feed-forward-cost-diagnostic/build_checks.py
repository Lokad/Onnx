"""Require unchanged product arithmetic, original Data flow and public contracts."""
from collections import Counter
import json

from il_check import same_except_markers

TENSOR = 'Lokad.Onnx.Tensor`1[T]::'
SINGLE = 'Lokad.Onnx.Tensor`1[System.Single]'
OPTIONS = 'Lokad.Onnx.TensorExecutionOptions'
WRAPPERS = {
    TENSOR + f'RunIsolatedShortWidePackedRows::Void RunIsolatedShortWidePackedRows(Int32, Int32, Int32, Single*, Single*, Single*, {OPTIONS})',
    TENSOR + f'RunWideProjectionMatMul2DCore::{SINGLE} RunWideProjectionMatMul2DCore({SINGLE}, {SINGLE}, Lokad.Onnx.DenseTensor`1[System.Single], {OPTIONS}, Boolean)',
    TENSOR + f'MatMulInto::{SINGLE} MatMulInto({SINGLE}, {SINGLE}, Lokad.Onnx.DenseTensor`1[System.Single], {OPTIONS}, Boolean)',
    TENSOR + f'MatMul::{SINGLE} MatMul({SINGLE}, {SINGLE}, {OPTIONS}, Lokad.Onnx.TensorBufferPool)',
    TENSOR + f'MatMul::{SINGLE} MatMul({SINGLE}, {SINGLE}, {OPTIONS})',
    TENSOR + f'RunPackedBatches::Void RunPackedBatches({SINGLE}, {SINGLE}, Int32[], Int32[], Int32[], Int32, Int32, Int32, Int32, Int32, Lokad.Onnx.DenseTensor`1[System.Single])',
    'Lokad.Onnx.Tensor`1+<>c__DisplayClass436_0[T]::<RunPackedBatches>b__0::Void <RunPackedBatches>b__0(Int32)',
}
PROFILE_START = 'Lokad.Onnx.ProfilerContext::StartNodeProfile::Void StartNodeProfile(Int64, Lokad.Onnx.OpType, System.Func`1[System.String])'
PROFILE_STOP = 'Lokad.Onnx.ProfilerContext::StopNodeProfile::Void StopNodeProfile()'
EXECUTE = ('Lokad.Onnx.ParakeetTranscriber::Execute::System.Collections.Generic.IReadOnlyDictionary`2[System.String,Lokad.Onnx.ITensor] '
           'Execute(Lokad.Onnx.GraphExecution, System.Collections.Generic.Dictionary`2[System.String,Lokad.Onnx.ITensor])')


def calls(value):
    body = json.loads(value) if isinstance(value, str) else value
    return Counter((i['opcode'], i['operand']) for i in body['instructions'] if i['opcode'] in ['call', 'callvirt'])


def clock_calls(before, after, start):
    old, new = calls(before), calls(after)
    assert not old - new, 'Original profiler call removed'
    expected = Counter({
        ('call', 'System.Diagnostics.Stopwatch::Int64 GetTimestamp()'): 1,
        ('call', 'Lokad.Onnx.ProfilerContext::System.Collections.Generic.List`1[Lokad.Onnx.WallNode] get_Wall()'): 2,
    })
    prefix = 'System.Collections.Generic.List`1[Lokad.Onnx.WallNode]::'
    extra = ['Void Add(Lokad.Onnx.WallNode)', 'Int32 get_Count()'] if start else [
        'Lokad.Onnx.WallNode get_Item(Int32)', 'Void set_Item(Int32, Lokad.Onnx.WallNode)']
    expected.update(('callvirt', prefix + name) for name in extra)
    assert new - old == expected, ('Unexpected profiler calls', new - old, expected)
    return dict(passed=True, extra_calls=sum(expected.values()), source_review_required=True)


def inventory(value, baseline, old_observer, products, built):
    assert value['inventory_complete'] and baseline['inventory_complete'] and old_observer['inventory_complete']
    assert len(value['observations']) == 2
    actual = {r['assembly']: r for r in value['observations']}
    expected = {r['assembly']: r for r in baseline['observations']}
    assert set(actual) == set(expected) == {'Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'}
    for name, row in actual.items():
        prior = expected[name]
        assert row['before_sha256'] == prior['before_sha256'] == products[name]['sha256']
        assert row['after_sha256'] == built['core' if name == 'Lokad.Onnx.dll' else 'data']['sha256']
        assert row['public_surface_equal'] and row['public_surface'] == prior['public_surface']
        assert row['compiler_rename'] is None and not row['removed']
        assert row['normalized_methods'] == prior['normalized_methods']
        assert row['method_flags_before'] == prior['method_flags_before']
        assert all(row['method_flags_after'][key] == flags for key, flags in row['method_flags_before'].items())
        assert set(row['candidate_methods']) == set(row['differences']) | set(row['added'])

    core = actual['Lokad.Onnx.dll']
    assert core['methods'] == 3254 and core['unchanged_methods'] == 3245 and not core['added']
    assert set(core['differences']) == WRAPPERS | {PROFILE_START, PROFILE_STOP}
    assert core['method_flags_before'] == core['method_flags_after']
    wrappers = {key: same_except_markers(core['normalized_methods'][key], core['candidate_methods'][key])
                for key in sorted(WRAPPERS)}
    profiles = {key: clock_calls(core['normalized_methods'][key], core['candidate_methods'][key], key == PROFILE_START)
                for key in [PROFILE_START, PROFILE_STOP]}
    data = actual['Lokad.Onnx.Data.dll']
    assert data['methods'] == 697 and data['unchanged_methods'] == 696 and data['differences'] == [EXECUTE]
    assert data['added'] and all(key.startswith(('Lokad.Onnx.ParakeetPhaseProbe', '<>f__AnonymousType')) for key in data['added'])
    observer = next(r for r in old_observer['observations'] if r['assembly'] == 'Lokad.Onnx.Data.dll')
    assert data['normalized_methods'] == observer['normalized_methods']
    assert data['candidate_methods'][EXECUTE] == observer['candidate_methods'][EXECUTE], 'Original using-scope flow changed'
    return dict(passed=True, core_methods=3254, unchanged_core_methods=3245,
        arithmetic_equivalent=True, wrapper_reviews=wrappers, profiler_reviews=profiles,
        original_data_methods=697, unchanged_data_methods=696,
        data_execute_matches_certified_observer=True, public_surface_equal=True, original_flags_equal=True)
