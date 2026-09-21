"""Audit complete carried trajectories and wall clocks independently of the consumer."""
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np
from protocol import ROOT, BASE, SITE, pin, read, save, verify, verify_recovery
from prepare import MANIFEST
OUTPUT = BASE/'trace-output'


def clocks(row, graph):
    assert row['frequency'] > 0 and row['reset_start_ticks'] <= row['reset_end_ticks'] <= row['start_ticks'] < row['end_ticks']
    expected = {n['id']: n for n in graph}; assert len(expected) == len(graph)
    if row['pass'] == 0:
        assert row['nodes'] == []
        return {}
    assert row['pass'] == 1 and len(row['nodes']) == len(graph)
    seen = set(); operators = defaultdict(Fraction); last = row['start_ticks']
    for node in row['nodes']:
        assert node['id'] not in seen and node['op'] == expected[node['id']]['op']; seen.add(node['id'])
        assert last <= node['start_ticks'] <= node['end_ticks'] <= row['end_ticks']; last = node['end_ticks']
        operators[node['op']] += Fraction(node['end_ticks']-node['start_ticks'], row['frequency'])
    assert seen == set(expected)
    return dict(operators)


def same(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes(), 'Tensor bits differ'


def array(value, dtype, shape):
    assert value.dtype == np.dtype(dtype) and value.shape == tuple(shape)
    assert np.isfinite(value).all()
    return value


def decode_text(tokens, pieces):
    text = ''.join(pieces[t] for t in tokens).replace('\u2581', ' ')
    return ''.join(c if not c.isspace() else ' ' if i > 0 and i+1 < len(text)
                   and (text[i+1] == '_' or text[i+1].isalnum()) else '' for i, c in enumerate(text))


def trajectory(calls, expected, pcm, pieces):
    assert len(calls) == expected['decoder_calls']+2
    front, encoder = calls[:2]; feature_frames = pcm.size//160+1; frames = (feature_frames+7)//8
    assert set(front['inputs']) == {'waveforms', 'waveforms_lens'} and set(front['outputs']) == {'features', 'features_lens'}
    same(array(front['inputs']['waveforms'], '<f4', [1, pcm.size]), pcm.reshape(1, -1))
    same(array(front['inputs']['waveforms_lens'], '<i8', [1]), np.array([pcm.size], dtype='<i8'))
    features = array(front['outputs']['features'], '<f4', [1, 128, feature_frames])
    same(array(front['outputs']['features_lens'], '<i8', [1]), np.array([feature_frames], dtype='<i8'))
    assert set(encoder['inputs']) == {'audio_signal', 'length'} and set(encoder['outputs']) == {'outputs', 'encoded_lengths'}
    same(encoder['inputs']['audio_signal'], features); same(encoder['inputs']['length'], front['outputs']['features_lens'])
    hidden = array(encoder['outputs']['outputs'], '<f4', [1, 1024, frames])
    same(array(encoder['outputs']['encoded_lengths'], '<i8', [1]), np.array([frames], dtype='<i8'))
    state1 = np.zeros((2, 1, 640), dtype='<f4'); state2 = state1.copy()
    tokens = []; positions = []; durations = []; frame = 0; emitted = 0
    for row in calls[2:]:
        assert frame < frames and len(tokens) < 4096
        i, o = row['inputs'], row['outputs']
        assert set(i) == {'encoder_outputs', 'targets', 'target_length', 'input_states_1', 'input_states_2'}
        assert set(o) == {'outputs', 'prednet_lengths', 'output_states_1', 'output_states_2'}
        same(array(i['encoder_outputs'], '<f4', [1, 1024, 1]), hidden[:, :, frame:frame+1])
        same(array(i['targets'], '<i4', [1, 1]), np.array([[tokens[-1] if tokens else 8192]], dtype='<i4'))
        same(array(i['target_length'], '<i4', [1]), np.array([1], dtype='<i4'))
        same(i['input_states_1'], state1); same(i['input_states_2'], state2)
        values = array(o['outputs'], '<f4', [1, 1, 1, 8198]).reshape(-1)
        same(array(o['prednet_lengths'], '<i4', [1]), np.array([1], dtype='<i4'))
        next1 = array(o['output_states_1'], '<f4', [2, 1, 640]); next2 = array(o['output_states_2'], '<f4', [2, 1, 640])
        token = int(np.argmax(values[:8193])); duration = int(np.argmax(values[8193:]))
        if token != 8192:
            state1 = next1; state2 = next2; tokens.append(token); positions.append(frame); durations.append(duration); emitted += 1
        if duration > 0: frame += duration; emitted = 0
        elif token == 8192 or emitted == 10: frame += 1; emitted = 0
    stop = 'EndOfAudio' if frame >= frames else 'TokenLimit'
    assert frame >= frames or len(tokens) == 4096
    actual = dict(text=decode_text(tokens, pieces), token_ids=tokens, frame_indices=positions, duration_frames=durations,
                  stop_reason=stop, encoded_frames=frames, decoder_calls=len(calls)-2)
    assert actual == expected, 'Complete greedy decisions differ'
    return actual


def main():
    sys.path.insert(0, str(SITE)); import psutil
    assert not (BASE/'analysis.json').exists()
    verify_recovery()
    frozen = read(BASE/'frozen.json'); verify(frozen['files'])
    assert frozen['prepared'] == pin(BASE/'prepared.json')
    states = [read(BASE/(mode+'-state.json')) for mode in ('trace', 'public')]; identities = []
    for state in states:
        assert state['complete'] and state['passed'] and state['code'] == 0
        for identity in (state['supervisor'], state['worker']):
            try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
            except psutil.NoSuchProcess: pass
            identities.append(identity)
    result = read(OUTPUT/'result.json'); public = read(BASE/'public-output/result.json')
    assert result['mode'] == 'trace' and result['applications'] == []
    assert public['mode'] == 'public' and public['call_files'] == [] and public['traced'] == []
    assert public['passed'] and public['error'] is None and public['inputs_and_held_outputs_unchanged']
    for key in ('runtime', 'affinity', 'processor_count', 'flags', 'manifest_sha256', 'core_sha256', 'data_sha256', 'runner_sha256'):
        assert public[key] == result[key]
    result['applications'] = public['applications']
    spec = read(MANIFEST); graphs = read(OUTPUT/'graphs.json')
    assert read(BASE/'public-output/graphs.json') == graphs
    assert result['passed'] and result['error'] is None and result['inputs_and_held_outputs_unchanged']
    assert result['runtime'] == '.NET 10.0.12' and result['affinity'] == 4 and result['processor_count'] == 1 and result['flags'] == {}
    assert result['manifest_sha256'] == pin(MANIFEST)['sha256']
    for key, filename in [('core_sha256', 'Lokad.Onnx.dll'), ('data_sha256', 'Lokad.Onnx.Data.dll'), ('runner_sha256', 'Profile.dll')]:
        assert result[key] == pin(BASE/'bin'/filename)['sha256']
    assert set(graphs) == {'frontend', 'encoder', 'decoder'}
    pieces = []
    for index, line in enumerate((ROOT/spec['models']['vocab.txt']['path']).read_text(encoding='utf-8-sig').splitlines()):
        piece, number = line.rsplit(' ', 1); assert int(number) == index; pieces.append(piece)
    assert len(pieces) == 8193 and pieces[-1] == '<blk>'
    calls = [read(OUTPUT/name) for name in result['call_files']]
    assert result['call_files'] == [f'{i:04}.json' for i in range(2480)]
    wanted = [(p, c['name'], graph, step) for p in range(2) for c in spec['cases']
              for graph, step in [('frontend', -1), ('encoder', -1)]+[('decoder', i) for i in range(c['expected']['decoder_calls'])]]
    assert [(r['pass'], r['name'], r['graph'], r['step']) for r in calls] == wanted
    assert [(r['pass'], r['name']) for r in result['traced']] == [(p, c['name']) for p in range(2) for c in spec['cases']]
    saved = {}; baseline = {}; values = 0; output_arrays = 0; input_arrays = 0
    def tensor(info):
        nonlocal values
        name = info['file']; path = (OUTPUT/name).resolve(); assert path.is_relative_to((OUTPUT/'arrays').resolve()) and name not in saved
        assert info['dtype'] in ('<f4', '<i4', '<i8') and all(isinstance(d, int) and d >= 0 for d in info['shape'])
        value = np.fromfile(path, dtype=info['dtype']); assert value.size == info['values'] == int(np.prod(info['shape']))
        assert np.isfinite(value).all() and pin(path)['sha256'] == info['sha256']
        saved[name] = pin(path); values += value.size
        return value.reshape(info['shape'])
    groups = defaultdict(list); by_graph = defaultdict(Fraction); by_op = defaultdict(Fraction); by_node = defaultdict(Fraction)
    per_clip = defaultdict(lambda: dict(seconds=Fraction(), operators=defaultdict(Fraction), resets=Fraction()))
    for row in calls:
        assert row['phase'] == ('unprofiled' if row['pass'] == 0 else 'wall') and row['allocated_bytes'] >= 0
        assert all(a <= b for a, b in zip(row['gc_before'], row['gc_after'], strict=True))
        operators = clocks(row, graphs[row['graph']]); key = row['name'], row['graph'], row['step']
        descriptors = {prefix+'/'+name: info for prefix in ('inputs', 'outputs') for name, info in row[prefix].items()}
        if row['pass'] == 0: baseline[key] = descriptors
        else:
            assert set(descriptors) == set(baseline[key])
            for name, info in descriptors.items():
                old = baseline[key][name]
                assert (old['dtype'], old['shape'], old['values']) == (info['dtype'], info['shape'], info['values'])
                assert (OUTPUT/old['file']).read_bytes() == (OUTPUT/info['file']).read_bytes(), (key, name)
            elapsed = Fraction(row['end_ticks']-row['start_ticks'], row['frequency'])
            by_graph[row['graph']] += elapsed; cell = per_clip[(row['name'], row['graph'])]; cell['seconds'] += elapsed
            cell['resets'] += Fraction(row['reset_end_ticks']-row['reset_start_ticks'], row['frequency'])
            for op, value in operators.items(): by_op[(row['graph'], op)] += value; cell['operators'][op] += value
            for n in row['nodes']: by_node[(row['graph'], n['id'])] += Fraction(n['end_ticks']-n['start_ticks'], row['frequency'])
        groups[(row['pass'], row['name'])].append(row)
    # Load only one complete request at a time; all captured values are checked.
    for pass_index in range(2):
        for case in spec['cases']:
            decoded = []
            for row in groups[(pass_index, case['name'])]:
                decoded.append({prefix: {name: tensor(info) for name, info in row[prefix].items()} for prefix in ('inputs', 'outputs')})
                input_arrays += len(row['inputs']); output_arrays += len(row['outputs'])
            pcm = np.load(ROOT/case['pcm']['path'], allow_pickle=False)
            actual = trajectory(decoded, case['expected'], pcm, pieces)
            observed = next(t['result'] for t in result['traced'] if t['pass'] == pass_index and t['name'] == case['name'])
            assert observed == actual
    assert (input_arrays, output_arrays) == (12160, 9760)
    public_seconds = Fraction()
    assert [r['name'] for r in result['applications']] == [c['name'] for c in spec['cases']]
    for row, case in zip(result['applications'], spec['cases'], strict=True):
        assert row['result'] == case['expected'] and row['frequency'] > 0 and row['end_ticks'] > row['start_ticks'] and row['allocated_bytes'] >= 0
        assert row['input_sha256'] == __import__('hashlib').sha256(np.load(ROOT/case['pcm']['path'], allow_pickle=False).tobytes()).hexdigest()
        public_seconds += Fraction(row['end_ticks']-row['start_ticks'], row['frequency'])
    expected_files = set(saved) | set(result['call_files']) | {'graphs.json', 'result.json'}
    assert {p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob('*') if p.is_file()} == expected_files
    assert sum((OUTPUT/n).stat().st_size for n in expected_files) <= 1024**3
    assert {p.relative_to(BASE/'public-output').as_posix() for p in (BASE/'public-output').rglob('*') if p.is_file()} == {'graphs.json', 'result.json'}
    sample_count = 0
    for state in states:
        samples = [json.loads(line) for line in (BASE/(state['mode']+'-samples.jsonl')).read_text().splitlines()]
        assert len(samples) == state['samples'] > 0 and max(s['rss'] for s in samples) == state['peak_rss']; sample_count += len(samples)
        assert state['preflight']['available'] >= (14 if state['mode'] == 'public' else 10)*1024**3 and state['preflight']['disk'] >= 20*1024**3
        for s in samples:
            assert s['seconds'] < 1800 and s['rss'] < (12 if state['mode'] == 'public' else 8)*1024**3 and s['available'] >= 1024**3 and s['disk'] >= 20*1024**3 and s['affinity'] == [2] and s['artifact_bytes'] <= 1024**3
        gaps = [b['seconds']-a['seconds'] for a, b in zip(samples, samples[1:])]; assert all(0 <= g < 10 for g in gaps)
    total = sum(by_graph.values()); metadata = {(g, n['id']): n for g, rows in graphs.items() for n in rows}
    profile = dict(passed=True, graph_calls=2480, decoder_calls=2400, graph_output_arrays=output_arrays, input_arrays=input_arrays,
        captured_values=values, traced_requests=40, public_requests=20, nodes_per_graph={g: len(rows) for g, rows in graphs.items()},
        graph_seconds={g: float(v) for g, v in by_graph.items()}, graph_shares={g: float(v/total) for g, v in by_graph.items()},
        operators=[dict(graph=g, op=op, seconds=float(v), share_of_graph=float(v/by_graph[g]), share_of_all_graphs=float(v/total)) for (g, op), v in sorted(by_op.items(), key=lambda item: item[1], reverse=True)],
        top_nodes=[dict(graph=g, **metadata[(g, identifier)], seconds=float(v), share_of_graph=float(v/by_graph[g])) for (g, identifier), v in sorted(by_node.items(), key=lambda item: item[1], reverse=True)[:30]],
        clips=[dict(name=name, graph=g, seconds=float(v['seconds']), reset_seconds=float(v['resets']), operators={k: float(n) for k, n in v['operators'].items()}, outside_nodes_seconds=float(v['seconds']-sum(v['operators'].values()))) for (name, g), v in per_clip.items()],
        separate_public_control_seconds=float(public_seconds), samples=sample_count, peak_rss=max(s['peak_rss'] for s in states),
        worker_resources={s['mode']: dict(samples=s['samples'], peak_rss=s['peak_rss']) for s in states},
        scope='Local full-corpus attribution; public controls and graph profiles have different instrumentation; no new matched native timing or tensor-native verdict')
    save(BASE/'analysis.json', profile)
    files = dict(frozen['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(BASE).parts): files[path.relative_to(ROOT).as_posix()] = pin(path)
    save(BASE/'closed.json', dict(passed=True, files=files, identities=identities, analysis=pin(BASE/'analysis.json')))
    print(json.dumps({k: v for k, v in profile.items() if k not in ('clips', 'top_nodes', 'operators')}))
    print(json.dumps(profile['operators'][:12])); print(json.dumps(dict(closure=pin(BASE/'closed.json'), files=len(files))))


if __name__ == '__main__': main()
