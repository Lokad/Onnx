"""Require direct completion and zero generic work for every original geometry."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1]/'depthwise-route-amd/expected.py'
loader = importlib.util.spec_from_file_location('old_depthwise_census', path)
old = importlib.util.module_from_spec(loader); loader.loader.exec_module(old)


def expected(diagnosis):
    rows = old.expected(diagnosis)
    for row in rows.values():
        row.pop('requested_scratch')
        row['direct_batches'] = row['calls']
        for name in ['tiled_batches', 'panels', 'products', 'views', 'patch_values']: row[name] = 0
        for name in ['panel_widths', 'matrix_shapes', 'leaves', 'scratch_elements']: row[name] = {}
    return rows


def audit_counts(report, diagnosis):
    target = expected(diagnosis)
    assert report['passed'] and report['protocol'] == 'parakeet-direct-depthwise-route-v1'
    assert report['runtime'] == '10.0.8' and report['processor_count'] == 1 and not report['flags']
    assert report['fma'] and report['avx2'] and report['avx512'] and report['vector_width'] == 8
    rows = {r['key']:r for r in report['rows']}; assert len(rows) == len(report['rows']) and rows.keys() == target.keys()
    for key, wanted in target.items():
        row = rows[key]
        for field, value in wanted.items(): assert row[field] == value, (key, field, row[field], value)
        assert row['simd'] and row['intrinsics'] and not row['segmented'] and row['degree'] == 1
        assert row['layouts'] and sum(row['layouts'].values()) == row['calls']
        assert all(count > 0 for count in row['layouts'].values())
        for layout in row['layouts']:
            for pair in layout.split(' | '):
                before, after = pair.split(' -> ')
                assert before == after and before.startswith('Lokad.Onnx.DenseTensor`1[System.Single]:False:')
    totals = {name:sum(r[name] for r in rows.values()) for name in
              ['calls', 'direct_batches', 'tiled_batches', 'completed_batches', 'panels', 'products', 'views', 'patch_values']}
    assert all(value % 4 == 0 for value in totals.values())
    return dict(passed=True, shapes=len(rows), totals=totals, per_corpus={k:v//4 for k,v in totals.items()},
                actual_route='successful direct-depthwise completion', every_geometry_exact=True,
                zero_generic_work=True, all_source_and_dense_layouts_equal=True)
