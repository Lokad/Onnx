"""Publish the closed complete-corpus observation without interpreting its clocks."""
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TOOLS = ROOT/'tests/parakeet/masking-padding-layout-amd'
sys.path.insert(0,str(TOOLS))
from run import BASE, pin, read, prepared
from review_build import resources
from audit_masking_layouts import correction


def main():
    prepared()
    proof = read(BASE/'closed.json'); assert proof['passed']
    for key,name in [('analysis','analysis.json'),('observations','observations.json'),
        ('build_review','build-review.json'),('collection','capture-collected/capture-collection.json'),
        ('transfer','capture-transfer.json')]:
        assert proof[key] == pin(BASE/name), name
    assert proof['auditor'] == pin(OUT/'audit_masking_layouts.py') and proof['checks'] == pin(TOOLS/'checks.py')
    assert proof['audit_correction'] == pin(BASE/'audit-correction.json')
    _, corrected = correction(); assert read(BASE/'audit-correction.json') == corrected
    analysis = read(BASE/'analysis.json'); observations = read(BASE/'observations.json')
    assert analysis['passed'] and analysis['core_unchanged'] and analysis['exact_public_results']
    assert analysis['diagnostic_only'] and analysis['no_performance_score']
    assert (analysis['requests'],analysis['observations'],len(observations)) == (80,9600,9600)
    assert analysis['resources'] == resources('capture')
    build = read(BASE/'build-review.json'); assert build['passed'] and build['core_unchanged']
    assert build['resources'] == resources('build')
    assert build['inventory'] == pin(BASE/'build-collected/inventory/instructions.json')
    assert build['built'] == pin(BASE/'build-collected/built.json')
    assert build['spec'] == pin(BASE/'bundle/spec.json')
    assert Counter(r['request'] for r in observations) == {i:120 for i in range(80)}
    masks = Counter((r['family'],r['mask']) for r in observations if r['mask'] is not None)
    assert analysis['masks'] == [dict(family=k[0],mask=k[1],observations=v) for k,v in sorted(masks.items())]
    requests = {i:read(BASE/'capture-collected/phase'/f'layout-{i:03}.json') for i in range(80)}
    raw_nodes = {i:{r['Name']:r for r in request['Calls'][0]['Records']} for i,request in requests.items()}
    assert all(len(nodes)==120 for nodes in raw_nodes.values())
    rows = []
    for observation in observations:
        request = requests[observation['request']]
        raw = raw_nodes[observation['request']][observation['name']]
        row = {k:v for k,v in observation.items() if k not in ['inputs','output']}
        row.update(clip=request['Name'],pass_index=request['Pass'])
        for index,value in [*enumerate(raw['Inputs']),('output',raw['Output'])]:
            row[str(index)] = json.dumps(value,separators=(',',':'),allow_nan=False)
        # Pad exports have optional operands; a shared CSV schema retains them.
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    lines = ['# Parakeet: actual masking and padding layouts','',
        '**All 80 complete requests and 9,600 target observations pass.**',
        'This is a diagnostic capture of the qualified composition, not a performance comparison.',
        'All public results exactly match the uninstrumented candidate, Core is unchanged,',
        'and original native-result, input and held-output checks remain in force.','',
        'The capture covers every one of the 20 clips, 24 encoder layers and 19 encoded',
        'lengths, with one warmup and three measured-label passes per clip. Each request',
        'contains 72 Where and 48 Pad observations. Those pass labels do not make these',
        'instrumented clocks suitable for a speedup claim.','',
        '| Mask family | Observed values | Observations |','|---|---|---:|']
    for row in analysis['masks']:
        lines.append(f"| {row['family']} | {row['mask']} | {row['observations']} |")
    lines += ['', 'The following table groups actual operand layouts; input numbers follow',
        'the ONNX node input order. Dense means the exact DenseTensor runtime type.',
        'Row-major and reversed describe the reported strides. Array offset is in',
        'elements; -1 denotes non-array-backed storage. Full dimensions, strides,',
        'backing lengths, scalar bits and padding values are retained in the CSV.','',
        '| Family | Operand | Dense | Row-major | Reversed | Array offset | Observations |',
        '|---|---|---|---|---|---:|---:|']
    for row in analysis['layouts']:
        v = row['layout']
        lines.append(f"| {row['family']} | {row['operand']} | {v['exact_dense']} | {v['row_major']} | {v['reversed']} | {v['array_offset']} | {row['observations']} |")
    lines += ['', 'The observer adds one disposable logging scope to the private Data graph-call',
        'helper and two consumer hooks. The compiled review preserves the original',
        'helper body, all other original methods, public interfaces and implementation',
        'flags. Metadata contains no retained activation storage; logging state is',
        'restored after each graph call. All build and capture owners are terminal.','',
        'The first local audit referenced the comparison result without its output/',
        'directory. A separate audit revision corrects that one retained-file path',
        'and binds the correction in the closure. Every original check remains;',
        'the capture, prepared tools and raw observations are unchanged. No inference reran.','',
        'Use these observed layouts with the [matched ORT work](masking-padding-20260924.md)',
        'and [source-derived work counts](masking-work-20260924.md) to select one bounded',
        'experiment. This capture does not establish native instruction counts, memory',
        'traffic or a candidate speedup. Previous failed screens retain their verdicts.','',
        '[All observations](masking-layouts-20260924.csv),',
        '[complete layout, mask and compiled review](masking-layouts-20260924.json),',
        '[observer protocol](../masking-padding-layout-amd/README.md).','',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    documents = {'masking-layouts-20260924.md':'\n'.join(lines)+'\n',
        'masking-layouts-20260924.csv':stream.getvalue(),
        'masking-layouts-20260924.json':json.dumps(dict(closure=pin(BASE/'closed.json'),
            analysis=analysis,build_review=build,audit_correction=corrected),indent=2,allow_nan=False)+'\n'}
    assert all(not (OUT/name).exists() for name in documents), 'Preserve existing publication'
    for name,content in documents.items():
        with (OUT/name).open('x',encoding='utf8',newline='') as target:target.write(content)
    print(json.dumps(dict(passed=True,requests=80,observations=len(rows),documents=list(documents))))


if __name__ == '__main__':main()
