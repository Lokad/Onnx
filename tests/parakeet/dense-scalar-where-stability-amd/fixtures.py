"""All 220 valid cases; the original 122 keep their exact order and partitions."""
from numerical_fixtures import cases
from original_fixtures import screen_cases as original_cases


def screen_cases(capture, qualified):
    result = original_cases(capture, qualified)
    old_names = {c['name'] for c in result}
    reference = {r['name']: r for r in qualified['rows']['provider']}
    census = cases(capture)
    for c in census:
        if c.get('error') or c['name'] in old_names: continue
        row = reference[c['name']]
        assert all(row[k] for k in ['oracle', 'inputs', 'held', 'owned'])
        result.append(dict(c, partition='added98', output_shape=row['shape'], expected_output=row['output'],
                           batch=max(1, min(1024, 65536 // max(1, row['values'])))))
    assert len(result) == len({c['name'] for c in result}) == 220
    assert {c['name'] for c in result} == {c['name'] for c in census if not c.get('error')}
    assert sum(c['partition']=='added98' for c in result) == 98
    return result
