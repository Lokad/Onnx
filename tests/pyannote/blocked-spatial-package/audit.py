"""Close public package execution, dependencies, product identity and resources."""
import json
from run import BASE, PRODUCT, pin, read, save, verify, terminal, package, prior


def main():
    assert not (BASE/'closed.json').exists(); prior()
    value = read(BASE/'verified.json'); assert value['passed']; verify(value['files'])
    assert package() == value['package'] == read(BASE/'package.json')
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == ['package', 'restore', 'build', 'consumer']
    observed = []; identities = [state['supervisor']]
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < 900
        assert row['preflight']['available'] >= (12 if row['name'] == 'consumer' else 8)*1024**3
        samples = [json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for s in samples:
            assert s['seconds'] < 900 and s['rss'] < 8*1024**3 and s['available'] >= 1024**3
            assert s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3 and s['rss'] == sum(m['rss'] for m in s['members'])
            for m in s['members']: assert m['affinity'] == [2] and row['members'][str(m['pid'])] == m['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        observed.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: terminal(identity)
    assert pin(BASE/'consumer.json') == value['result']
    result = read(BASE/'consumer.json'); assert result['passed'] and result['pid'] == state['runs'][-1]['worker']['pid']
    assert result['core'] == value['core']['sha256'] and result['executable'] == value['consumer']['sha256']
    assert result['prepared_graph_calls'] == 2 and result['prepared_graph_values'] == 1056
    assert result['retained_weights'] == 18432 and result['graph_scratch'] == 8384
    assert result['model_imported'] and result['input_and_held_outputs_unchanged'] and result['processor_count'] == 1
    analysis = dict(passed=True, core=value['core'], package=value['package']['package'], dependencies=value['package']['dependencies'],
        prepared_graph_values=1056, graph_calls=2, retained_weights=18432, requested_scratch=8384,
        resources=observed, no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=value['files'], identities=identities, analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()
