"""Audit the actual graph intervention and every retained numerical boundary."""
from experiment import *


def metric(actual, reference):
    assert actual.shape == reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all()
    delta = actual.astype(np.float64)-reference.astype(np.float64)
    scaled = np.abs(delta)/np.maximum(1., np.abs(reference.astype(np.float64)))
    maximum = float(scaled.max()); failed = int(np.count_nonzero(scaled > 1e-4))
    # Recompute in independently traversed chunks, including awkward tail sizes.
    second_max = 0.; second_failed = 0
    for offset in range(0, actual.size, 65537):
        a = actual.ravel()[offset:offset+65537].astype(np.float64)
        b = reference.ravel()[offset:offset+65537].astype(np.float64)
        errors = np.abs(a-b)/np.where(np.abs(b) > 1., np.abs(b), 1.)
        second_max = max(second_max, float(np.max(errors)))
        second_failed += int(np.sum(errors > .0001))
    assert maximum == second_max and failed == second_failed
    return dict(values=actual.size, max_scaled=maximum, failed_values=failed,
                rms=float(np.sqrt(np.mean(delta*delta))), bitwise=actual.dtype == reference.dtype and actual.tobytes() == reference.tobytes())


def graph_change(base, variant, model):
    assert len({n['name'] for n in base}) == len(base) and len({n['name'] for n in variant}) == len(variant)
    original = {n.name:n for n in model.graph.node}
    producer = {out:n for n in model.graph.node for out in n.output}
    restored = set(); fused = set(); chains = []
    nodes = list(model.graph.node)
    for index, node in enumerate(nodes):
        if node.op_type != 'Erf':
            continue
        chain = nodes[index-1:index+4]
        assert [n.op_type for n in chain] == ['Div', 'Erf', 'Add', 'Mul', 'Mul']
        bias = producer[chain[0].input[0]]
        if bias.op_type == 'Add':
            chain = [bias]+chain
            fused.add(bias.name)
        else:
            assert bias.op_type == 'Conv'
            fused.add(chain[-1].name)
        restored.update(n.name for n in chain)
        chains.append([n.name for n in chain])
    assert len(chains) == len(fused) == 34 and len(restored) == 202
    actual_fusions = {n['name'] for n in base if n['op'] in ['Gelu', 'BiasGelu', 'GemmGelu']}
    assert actual_fusions == fused
    assert [n for n in base if n['name'] not in fused] == [n for n in variant if n['name'] not in restored], 'Non-GELU optimizer changes'
    restored_nodes = [n for n in variant if n['name'] in restored]
    assert len(restored_nodes) == len(restored)
    for node in restored_nodes:
        old = original[node['name']]
        assert node['op'] == old.op_type and node['inputs'] == list(old.input) and node['outputs'] == list(old.output)
        assert not node['fused'] and not old.attribute and not node['attributes']
    assert len(base) == 871 and len(variant) == 1039
    return dict(baseline_nodes=len(base), unfused_nodes=len(variant), unchanged_nodes=len(base)-len(fused), restored_nodes=len(restored), chains=chains)


def load(path, shape, dtype='<f4'):
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    assert np.isfinite(value).all()
    return value


def output_folder(spec, job_id):
    artifact = ROOT/spec['reused']['artifact'] if job_id == '00-baseline' else BASE
    return artifact/'outputs'/job_id


def main():
    spec = read(BASE/'manifest.json'); assert spec['protocol'] == PROTOCOL and spec['limits'] == LIMITS
    for name, expected in spec['files'].items():
        assert pin(ROOT/name) == expected, name
    for name, expected in spec['numerical_files'].items():
        assert pin(name) == expected, name
    state = read(BASE/'processes.json')
    assert state['complete'] and state['code'] == 0 and state['manifest'] == pin(BASE/'manifest.json')
    assert absent(state['supervisor']) and [r['job'] for r in state['runs']] == spec['jobs']
    assert all(a['ended'] <= b['started'] for a,b in zip(state['runs'], state['runs'][1:]))
    model = onnx.load(ROOT/spec['models']['baseline']['file'], load_external_data=False)
    variant = onnx.load(ROOT/spec['models']['unfused']['file'], load_external_data=False)
    assert append_erf_outputs(model).SerializeToString(deterministic=True) == variant.SerializeToString(deterministic=True)
    results = {}; rows = []; bridges = []; native = []; contrasts = []; resource = []; repeat = {}; graphs = {}; count = 0
    all_runs = [spec['reused']['run']]+state['runs']
    for run in all_runs:
        job = run['job']; reused = job['id'] == '00-baseline'
        artifact = ROOT/spec['reused']['artifact'] if reused else BASE
        assert run['complete'] and run['code'] == 0 and absent(run['worker'])
        assert run['preflight']['available'] >= LIMITS['preflight_available'] and run['preflight']['disk'] >= LIMITS['disk']
        if not reused:
            checks = run['preflight_checks']
            assert checks and checks[-1]['seconds'] <= spec['preflight_wait_seconds']+2
            assert {k:checks[-1][k] for k in ['available','disk']} == run['preflight']
            assert all(c['available'] < LIMITS['preflight_available'] or c['disk'] < LIMITS['disk'] for c in checks[:-1])
        samples = [json.loads(line) for line in (artifact/'process'/job['id']/'samples.jsonl').read_text().splitlines()]
        assert len(samples) == run['samples'] > 0
        for sample in samples:
            assert sample['pid'] == run['worker']['pid'] and sample['birth'] == run['worker']['birth']
            assert sample['seconds'] < LIMITS['seconds'] and sample['rss'] < LIMITS['rss']
            assert sample['available'] >= LIMITS['available'] and sample['disk'] >= LIMITS['disk'] and sample['affinity'] == [2]
        assert run['peak_rss'] == max(s['rss'] for s in samples)
        resource.append(dict(job=job['id'], samples=len(samples), peak_rss=run['peak_rss']))
        folder = output_folder(spec, job['id']); result = read(folder/'result.json'); mode = job['mode']; request = spec['requests'][job['request']]
        assert result['job'] == job and result['complete'] and result['inputs_unchanged'] and result['held_outputs_unchanged_after_reset']
        expected_manifest = spec['reused']['manifest_sha256'] if reused else pin(BASE/'manifest.json')['sha256']
        expected_probe = spec['reused']['probe_sha256'] if reused else pin(BASE/'bin/WhisperManagedGeluV2.dll')['sha256']
        assert result['manifest_sha256'] == expected_manifest and result['core_sha256'] == CORE
        assert result['probe_sha256'] == expected_probe
        assert result['runtime'] == '10.0.12' and result['affinity'] == 4 and result['processor_count'] == 1 and result['flags'] == {}
        assert not result['native_loaded'] and result['packed_weight_bytes'] == 256*1024**2
        assert result['input_sha256'] == request['input']['raw_sha256']
        assert result['nodes_sha256'] == pin(folder/'nodes.json')['sha256']
        nodes = read(folder/'nodes.json')
        assert len(nodes) == result['optimized_nodes'] and dict(collections.Counter(n['op'] for n in nodes)) == result['node_census']
        if mode in graphs:
            assert nodes == graphs[mode]
        graphs[mode] = nodes
        descriptions = spec['models'][mode]['outputs']
        assert len(result['outputs']) == len(descriptions)
        expected_files = {'result.json', 'nodes.json'}
        for index, (record, desc) in enumerate(zip(result['outputs'], descriptions, strict=True)):
            assert record['index'] == index and record['name'] == desc['name'] and record['shape'] == desc['shape']
            assert record['values'] == math.prod(desc['shape']) and record['file'] == f'{index:02}.f32'
            path = folder/record['file']; expected_files.add(path.name)
            assert pin(path) == dict(bytes=record['values']*4, sha256=record['sha256'])
            actual = load(path, desc['shape']); count += 1
            if job['request'] == 0:
                repeat[mode, index] = record['sha256']
            if job['request'] == 3:
                assert record['sha256'] == repeat[mode, index], 'Repeat bytes changed'
            if index >= 41:
                continue
            for engine in ['numpy', 'ort']:
                ref = request['references'][engine][index]
                assert ref['shape'] == desc['shape'] and ref['name'] == desc['name']
                value = metric(actual, load(ROOT/ref['file'], desc['shape'], '<f8'))
                rows.append(dict(request=job['request'], name=request['name'], mode=mode, reference=engine, boundary=index, **value))
            if mode == 'baseline':
                old = request['old_trace'][index]
                assert old['shape'] == desc['shape'] and old['name'] == desc['name']
                bridges.append(dict(request=job['request'], boundary=index, **metric(actual, load(ROOT/old['file'], desc['shape']))))
            else:
                contrasts.append(dict(request=job['request'], boundary=index,
                    **metric(actual, load(output_folder(spec, f"{job['request']:02}-baseline")/f'{index:02}.f32', desc['shape']))))
            if index == 40:
                for engine, key in [('managed', 'MM'), ('native', 'NM')]:
                    old = request['baselines'][key]; assert old['format'] == 'f32'
                    native.append(dict(request=job['request'], mode=mode, original_engine=engine,
                        **metric(actual, load(ROOT/old['file'], desc['shape']))))
        assert {p.name for p in folder.iterdir()} == expected_files
        results[job['id']] = result
    assert count == 592 and len(rows) == 656 and len(bridges) == len(contrasts) == 164
    intervention = graph_change(graphs['baseline'], graphs['unfused'], model)
    final = [r for r in rows if r['boundary'] == 40]
    final_map = {(r['request'],r['mode'],r['reference']):r for r in final}
    useful = all(final_map[i,'unfused',engine]['max_scaled'] <= .5*final_map[i,'baseline',engine]['max_scaled']
                 for i in range(3) for engine in ['numpy','ort'])
    observations = dict(structural_passed=True, useful_signal=useful, strict_final_passed=all(r['failed_values'] == 0 for r in final if r['mode'] == 'unfused'),
        manifest=pin(BASE/'manifest.json'), calls=7, reused_calls=1, arrays=count, comparisons=rows, final=final, bridges=bridges, contrasts=contrasts,
        native=native, resources=resource, intervention=intervention, censuses={k:dict(collections.Counter(n['op'] for n in v)) for k,v in graphs.items()})
    write(BASE/'observations.json', observations)
    report = Path(__file__).with_name('results-20260921.md')
    lines = ['# Actual managed Whisper GELU fusion intervention', '',
        'Seven fresh managed encoder calls and one verified reused baseline retain',
        'all 592 complete output arrays. The diagnostic graph adds the 34 original Erf',
        'outputs and 32 feed-forward activation outputs; all original nodes,',
        'constants and external weights are unchanged. The qualified current core is',
        '`'+CORE+'` (product `0f86c5d`).', '',
        f"The fixed requirement to halve final maximum error on all three unique clips",
        f"against both saved double references **{'passes' if useful else 'fails'}**. The unchanged",
        f"strict `1e-4` final-output gate **{'passes' if observations['strict_final_passed'] else 'fails'}**.", '',
        '| Clip | Mode | Maximum scaled error | Values above 1e-4 |',
        '|---|---|---:|---:|']
    for row in final:
        if row['request'] < 3 and row['reference'] == 'numpy':
            lines.append(f"| {row['name']} | {row['mode']} | {row['max_scaled']:.9g} | {row['failed_values']} |")
    identical = sum(r['bitwise'] for r in bridges)
    lines += ['', 'The table uses the independent NumPy double reference; the retained JSON',
        'also includes the independent ORT-double/math.erf route, all 41 common',
        'boundaries, original native final-output contrasts, and the exact repeated clip.', '',
        f"The actual optimized graph changes from {intervention['baseline_nodes']} to {intervention['unfused_nodes']} nodes:",
        f"34 fused activations expand to {intervention['restored_nodes']} original nodes, while all",
        f"{intervention['unchanged_nodes']} remaining nodes retain identical operations, inputs, outputs and attributes.",
        f"The fresh baseline matches {identical}/164 saved older managed trace arrays bit for bit.", '',
        'Every input remains unchanged, held outputs survive context reset, and the',
        'repeated clip reproduces all baseline and diagnostic arrays exactly. All',
        f"{sum(r['samples'] for r in resource)} resource samples pass; maximum worker RSS is",
        f"{max(r['peak_rss'] for r in resource):,} bytes. All original and new process births are terminal.", '',
        'This is a local Windows i7-14700KF CPU2 numerical experiment with .NET 10.0.12,',
        'Memory execution, a 256 MiB packed-weight budget and no runtime overrides.',
        'Additional observable outputs change lifetimes, so elapsed time is not a',
        'performance comparison. The intervention changes the complete GELU fusion',
        'and dependent bias fusion; it does not isolate reciprocal rounding alone.',
        'No production code, default, numerical tolerance or VM workload changed.', '',
        'The [first design](../managed-gelu/failure-20260921.md) remains a failed',
        'experiment: exposing Erf alone allowed 32 ScaledMatMul fusions, and the',
        'next worker was refused by the unchanged 10 GiB preflight memory reserve.',
        'This corrected graph exposes the final activation values too, preventing',
        'those matrix fusions. Its fixed limits and numerical screen are unchanged.', '',
        'Before starting a worker, the corrected supervisor may wait up to fifteen',
        'minutes for the same preflight reserve; every observation is retained.', '',
        'Artifact: `artifacts/whisper-managed-gelu-v2-20260921`.',
        'The frozen manifest, complete arrays, optimized node descriptions, resource',
        'samples and closure receipt retain the evidence. See [usage](README.md).', '']
    with report.open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines))
    public = Path(__file__).with_name('observations-20260921.json')
    write(public, observations)
    births = spec['reused']['births']+[state['supervisor']]+[r['worker'] for r in state['runs']]
    assert all(absent(identity) for identity in births)
    files = {p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    write(BASE/'closed.json', dict(structural_passed=True, useful_signal=useful, files=files, births=births,
        reports={rel(p):pin(p) for p in [report, public]}))
    print(json.dumps(dict(structural_passed=True, useful_signal=useful, strict_final_passed=observations['strict_final_passed'],
        arrays=count, final=final, closure=pin(BASE/'closed.json')), indent=2))


if __name__ == '__main__':
    main()
