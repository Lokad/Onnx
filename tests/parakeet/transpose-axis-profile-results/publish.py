"""Publish complete attribution only after both fresh captures pass their audits."""
import json
from pathlib import Path
from partition import ROOT, TOOLS, MAPPING, pin, read, references, partition
from run import BASE, APP, PARENT, native, prepared, write

RESULT = ROOT/'artifacts/parakeet-transpose-axis-gap-20260928'


def main():
    assert not RESULT.exists()
    outputs = [TOOLS/name for name in ['observations-20260928.json','diagnosis-20260928.md']]
    assert not any(path.exists() for path in outputs)
    context = native.qualification()
    prepared()
    proof = read(BASE/'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(BASE/'analysis.json')
    for name, wanted in proof['files'].items():
        assert pin(BASE/name) == wanted, name
    managed = read(BASE/'analysis.json'); spec = read(BASE/'bundle/spec.json')
    assert managed['attribution_only'] and managed['complete_admitted_public_results_exact']
    assert spec['qualification_closures']['root'] == context['root']
    assert spec['reference_product'] == context['measured']
    assert {name:spec[key] for name,key in [('Lokad.Onnx.dll','core'),('Lokad.Onnx.Data.dll','data')]} == context['built']
    closure = read(native.BASE/'closed.json')
    binding = read(native.BASE/'context-closed.json')
    assert binding['passed'] and binding['original_attribution'] == pin(native.BASE/'closed.json')
    assert binding['qualification'] == pin(native.BASE/'qualification.json')
    assert binding['adapter'] == pin(native.TOOLS/'run.py') and binding['auditor'] == pin(native.TOOLS/'analyze.py')
    assert read(native.BASE/'qualification.json') == context
    assert closure['passed'] and closure['analysis'] == pin(native.BASE/'analysis.json')
    assert closure['transfer'] == pin(native.BASE/'transfer.json')
    receipt = native.BASE/'collected/collection.json'
    assert closure['collection'] == pin(receipt)
    for name, wanted in read(receipt)['files'].items():
        assert pin(native.BASE/'collected'/name) == wanted, name
    ort = read(native.BASE/'analysis.json')
    assert ort['attribution_only'] and not ort['benchmark_update']
    native_spec = read(native.BASE/'collected/spec.json')
    assert native_spec['app'] == spec['app'] == native.REMOTE_APP
    manifest = pin(APP/'collected/manifests/current-parakeet.json')
    assert native_spec['manifest'] == spec['external'][spec['app']+'/manifests/current-parakeet.json'] == manifest
    mapping, old_managed, old_native = references()
    rows = partition(managed, ort, mapping, old_managed, old_native)
    value = dict(passed=True, managed_profile=pin(BASE/'closed.json'),
        native_profile=pin(native.BASE/'closed.json'), native_context=pin(native.BASE/'context-closed.json'),
        root=context['root'], application=context['application'], product=context['built'],
        measured_product=context['measured'], manifest=manifest,
        mapping_closure=pin(MAPPING/'closed.json'), mapping_analysis=pin(MAPPING/'analysis.json'),
        partition=rows, managed_corpus=managed['corpus'],
        native_corpus={mode:report['corpus_seconds'] for mode,report in ort['phases'].items()},
        phase_over_control=managed['phase_over_control'], wall_over_phase=managed['wall_over_phase'],
        native_control_over_unprofiled=ort['control_over_original'], native_profile_over_control=ort['profile_over_control'],
        all_managed_descriptors_exact=True, all_native_nodes_shapes_and_calls_exact=True,
        nonoverlapping_complete_partition=True, historical_clocks_used=False,
        attribution_only=True, fresh_cross_engine_score=False, overhead_subtracted=False,
        selected_optimization=None,
        helpers={str(path.relative_to(ROOT)):pin(path) for path in [TOOLS/'partition.py', PARENT/'diagnose.py']})
    lines = ['# Current Parakeet: managed and ORT attribution', '',
        'Both profiles use the original twenty-clip application and graph outputs.',
        'The actual qualified product and retained reviewed observers are bound to',
        'the same admitted application. Every managed public result remains exact.', '',
        '| Engine | Control seconds | Phase seconds | Node-profile seconds |', '|---|---:|---:|---:|',
        f"| Lokad | {managed['corpus']['control']:.6f} | {managed['corpus']['phase']:.6f} | {managed['corpus']['wall']:.6f} |",
        f"| Microsoft ORT | {ort['phases']['control']['corpus_seconds']:.6f} | — | {ort['phases']['profile']['corpus_seconds']:.6f} |", '',
        f"Managed phase/control is {value['phase_over_control']:.6f}; wall/phase is {value['wall_over_phase']:.6f}.",
        f"ORT control/unprofiled is {value['native_control_over_unprofiled']:.6f}; profile/control is {value['native_profile_over_control']:.6f}.",
        'These observations are diagnostic. No overhead is subtracted and no',
        'profiler clock updates BENCHMARK.md.', '',
        '| Work | Managed seconds | ORT seconds | Difference |', '|---|---:|---:|---:|']
    lines += [f"| {r['group']} | {r['managed_seconds']:.6f} | {r['ort_seconds']:.6f} | {r['excess_seconds']:.6f} |" for r in rows]
    lines += ['', 'All 2,856 managed and 1,993 ORT encoder nodes are counted once.',
        'The remaining rows include all other graph time and time outside graphs.',
        'Managed descriptors and native nodes, runtime shapes and call counts',
        'match the reviewed mapping. Every timing comes from these fresh captures.',
        'Historical timings contribute nothing to this partition.', '',
        'The differences identify work to investigate. They do not prove a kernel',
        'dispatch, attribute the cause, or promise additive application savings.',
        'Inspect the applicable ORT implementation before selecting one mechanism.', '',
        '[Complete membership, clocks, overhead and provenance](observations-20260928.json).',
        '[Independent unprofiled application comparison](../transpose-axis-results/application-20260928.md).']
    RESULT.mkdir()
    write(RESULT/'analysis.json', value)
    write(RESULT/'closed.json', dict(passed=True, attribution_only=True,
        analysis=pin(RESULT/'analysis.json'), reviewer=pin(Path(__file__))))
    write(outputs[0], dict(closure=pin(RESULT/'closed.json'), **value))
    lines += ['', 'Closure: `' + pin(RESULT/'closed.json')['sha256'] + '`.']
    with outputs[1].open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True, closure=pin(RESULT/'closed.json'),
        partition=[{k:v for k,v in row.items() if not k.endswith('_members')} for row in rows])))


if __name__ == '__main__':
    main()
