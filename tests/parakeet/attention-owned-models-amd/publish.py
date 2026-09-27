"""Publish the completed original model audit without inference or rescoring."""
import json
from pathlib import Path
from prepare import BASE,ROOT
from protocol import pin,read


def main():
    proof=read(BASE/'closed.json')
    assert pin(BASE/'closed.json')['sha256']=='2d64fad91c26c00ffe7cd003aa128efc4bc92280da96a9bb805a8e2b93412177'
    assert proof['passed'] and proof['analysis']==pin(BASE/'analysis.json')
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json')
    assert analysis['passed'] and len(analysis['results'])==8
    for role in ['selected','candidate']:
        for mode in ['512','256']:
            native=analysis['results'][f'{role}-native-{mode}']['native']
            public=analysis['results'][f'{role}-public-{mode}']
            assert (native['arrays'],native['values'])==(784,3090494)
            assert native['numeric_gate_passed'] and not native['failures']
            assert public['passed'] and public['public_requests']==20
            if role=='candidate':
                assert len(native['exact_selected_comparisons'])==784
                assert all(r['bit_identical'] for r in native['exact_selected_comparisons'])
                assert public['complete_selected_results_exact']
    target=ROOT/'tests/parakeet/attention-owned-results/models-20260928.json'
    report=dict(model_closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),
        observations=analysis,application_scored=False,release_admitted=False,source=pin(Path(__file__)))
    with target.open('x',encoding='utf8') as f:
        json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(published=pin(target),arrays=3136,values=12361976,public_requests=80)))


if __name__=='__main__':main()
