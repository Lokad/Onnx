"""Publish complete-model evidence without running inference or scoring time."""
import json
from pathlib import Path
from prepare import BASE,ROOT
from protocol import pin,read

assert pin(BASE/'closed.json')['sha256'] == 'f773277daa848121c199c58e67bce3777677c2ee2654fbe26250d02719155960'
proof = read(BASE/'closed.json'); assert proof['passed']
for name,wanted in proof['files'].items(): assert pin(BASE/name) == wanted,name
analysis = read(BASE/'analysis.json'); native = {}; public = {}
for name,value in analysis['results'].items():
    assert value['passed'] and value['no_performance_measurement']
    if 'native' in value:
        result = value['native']
        native[name] = {k:result[k] for k in ['arrays','values','maximum','audit_consistent','application_passed','numeric_gate_passed','failures']}
        if name.startswith('candidate-'):
            comparisons = result['exact_selected_comparisons']
            assert len(comparisons) == 784 and all(c['bit_identical'] for c in comparisons)
            native[name]['exact_current_arrays'] = len(comparisons)
    else:
        public[name] = value
report = dict(passed=True,performance_measured=False,model_closure=pin(BASE/'closed.json'),
    full_analysis=pin(BASE/'analysis.json'),identities=analysis['identities'],consumers=analysis['consumers'],
    arrays=sum(r['arrays'] for r in native.values()),values=sum(r['values'] for r in native.values()),
    public_requests=sum(r['public_requests'] for r in public.values()),native=native,public=public,
    resources=analysis['resources'],source=pin(Path(__file__)))
assert (report['arrays'],report['values'],report['public_requests']) == (3136,12361976,80)
target = ROOT/'tests/parakeet/decoder-lstm-layout-profile-results/pointwise-tail-models-observations-20260927.json'
with target.open('x',encoding='utf8') as stream: json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
print(json.dumps(dict(passed=True,published=pin(target),arrays=report['arrays'],values=report['values'],public_requests=80)))
