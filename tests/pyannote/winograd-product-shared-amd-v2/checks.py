"""Recompute the complete original shared/e5 scope and original native limits and exact unaffected same-platform bits."""
import numpy as np
from protocol import pin, read

CASES = ['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
METADATA = ['model','scenario','step','name','shape','values','reference_file','reference_sha256']


def compare_arrays(actual, wanted, selected, affected):
    assert actual.dtype == wanted.dtype == np.float32 and actual.shape == wanted.shape
    assert np.isfinite(actual).all() and np.isfinite(wanted).all()
    delta=np.abs(actual.astype(np.float64)-wanted.astype(np.float64))/np.maximum(1.,np.abs(wanted.astype(np.float64)))
    maximum=float(delta.max(initial=0));assert not np.count_nonzero(delta > 1e-4), 'Original native bound'
    exact=None
    if selected is not None:
        assert selected.dtype == np.float32 and selected.shape == actual.shape and np.isfinite(selected).all()
        exact=actual.tobytes()==selected.tobytes()
        assert affected or exact, 'Changed unaffected model bits'
    return maximum,exact


def expected_rows(base, mode):
    if mode == 'e5':
        return [(case, policy+'-'+context, step, 'last_hidden_state', read(base/'e5'/(case+'.json'))['reference_file'])
                for case in CASES for policy in ['default','memory'] for context in ['facade','context'] for step in range(3)]
    return [(model['key'], scenario['name'], step, value['name'], value['file'])
            for model in read(base/'reference/manifest.json')['models'] for scenario in model['scenarios']
            for step, item in enumerate(scenario['steps']) for value in item['outputs']]


def qualify(base, name, spec):
    role, mode = name.split('-'); assert role in ['selected','candidate'] and mode in ['shared','e5']
    folder = base/name/'output'; result = read(folder/'result.json')
    assert result['passed'] and result['mode'] == mode and result['enabled'] is False
    assert result['core_sha256'] == spec['identities'][role]['Lokad.Onnx.dll']['sha256']
    assert result['probe_sha256'] == spec['consumer']['sha256'] and result['runtime'] == spec['runtime'] == '10.0.8'
    assert result['flags'] == dict(LOKAD_ONNX_FINGERPRINT_STRINGS='0')
    assert result['inputs_unchanged'] and result['held_outputs_unchanged']
    assert len(result['graphs']) == (80 if mode == 'e5' else 11) and all(g['entries'] == 0 for g in result['graphs'])
    assert [(r['model'], r['scenario'], r['step'], r['name'], r['reference_file']) for r in result['rows']] == expected_rows(base, mode)
    historical = read(base/'evidence'/(mode+'-historical.json'))
    reference = base/('e5' if mode == 'e5' else 'reference')
    assert {p.name for p in folder.iterdir()} == {'result.json'} | {str(i)+'.f32' for i in range(len(result['rows']))}
    selected = read(base/('selected-'+mode)/'output/result.json') if role == 'candidate' else None
    rows = []
    for index, (row, old) in enumerate(zip(result['rows'], historical['rows'], strict=True)):
        assert all(row[k] == old[k] for k in METADATA)
        path = folder/row['file']; assert row['file'] == str(index)+'.f32'
        assert pin(path) == dict(bytes=row['values']*4, sha256=row['sha256'])
        native = reference/row['reference_file']; assert pin(native)['sha256'] == row['reference_sha256']
        actual = np.fromfile(path, dtype='<f4')
        wanted = np.fromfile(native, dtype='<f4') if mode == 'e5' else np.load(native, allow_pickle=False)
        assert actual.dtype == wanted.dtype == np.float32 and np.isfinite(actual).all() and np.isfinite(wanted).all()
        assert actual.size == wanted.size == row['values'] == int(np.prod(row['shape']))
        assert row['shape'] == (read(base/'e5'/(row['model']+'.json'))['shape'] if mode == 'e5' else list(wanted.shape))
        selected_array=None
        if selected is not None:
            other = selected['rows'][index]; assert all(row[k] == other[k] for k in METADATA)
            selected_array=np.fromfile(base/('selected-'+mode)/'output'/other['file'],dtype='<f4')
        maximum,exact=compare_arrays(actual,wanted.reshape(-1),selected_array,
            row['model'] in spec['arithmetic_scope']['models_with_eligible_convolutions'])
        assert row['failed_values'] == 0 and abs(maximum-row['max_scaled_error']) <= 1e-15
        rows.append(dict(model=row['model'], scenario=row['scenario'], step=row['step'], name=row['name'],
            values=row['values'], maximum=maximum, exact_selected=exact if selected is not None else None,
            arithmetic_changed_model=row['model'] in spec['arithmetic_scope']['models_with_eligible_convolutions']))
    assert len(rows) == (60 if mode == 'e5' else 106)
    return dict(passed=True, arrays=len(rows), values=sum(r['values'] for r in rows), rows=rows, no_performance_measurement=True)
