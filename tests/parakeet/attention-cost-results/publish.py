"""Publish the completed attention cost observation without running inference."""
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'attention-cost-amd'))
from run import BASE, QUALIFIED, prepared, pin, read, write
from partition import partition


def main():
    prepared()
    assert pin(BASE/'closed.json')['sha256'] == '5ff17ed5e3e9596789bf27b27abc01584342b08f81005a4a25643b0df045b5f2'
    closed = read(BASE/'closed.json')
    assert closed['passed'] and closed['usable_for_candidate_selection']
    for name,wanted in closed['files'].items(): assert pin(BASE/name) == wanted,name
    assert pin(BASE/'analysis.json') == closed['analysis']
    analysis = read(BASE/'analysis.json')
    folder = BASE/'capture-collected'
    costs = read(folder/'logs/costs.json')
    records = read(folder/'probe/observed/result.json')['records']
    expected = read(folder/'expected-attention.json')
    assert partition(costs,records,read(folder/'manifest.json'),expected) == analysis['groups']
    metadata = {(r['request'],r['weight']):r for r in expected['rows']}
    rows = []
    for row in costs['rows']:
        index = row['index']//120
        meta = metadata[index,row['weight']]
        value = {k:meta[k] for k in ['request','clip','pass_index','phase','node','weight','m','route']}
        value.update({k:row[k] for k in ['index','main_rows','mapped','row_guard_allows','fallback','leaf',
            'thread','scratch_elements','input_layout','weight_layout','start_ticks','end_ticks']})
        for i,stage in enumerate(costs['stages']):
            for key in ['ticks','calls','starts','ends']:
                value[stage+'_'+key] = row['stage_'+key][i]
        rows.append(value)
    clocks = HERE/'clocks-20260927.csv'
    observation = HERE/'observations-20260927.json'
    assert not clocks.exists() and not observation.exists()
    with clocks.open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(observation,dict(closure=pin(BASE/'closed.json'),qualified_root=pin(QUALIFIED/'closed.json'),
        compiled_review=pin(BASE/'build-review.json'),observer_product=read(folder/'built.json')['product'],
        source=pin(BASE/'bundle/spec.json'),route_binding=expected['closure'],clocks=pin(clocks),
        publisher=pin(Path(__file__)),observations=len(rows),frequency=costs['frequency'],
        overhead_subtracted=False,benchmark_updated=False,**analysis))
    print(json.dumps(dict(published=True,observations=len(rows),clocks=pin(clocks),summary=pin(observation))))


if __name__ == '__main__': main()
