"""Record the complete expanded reduction, spanning five fall-through JIT blocks."""
from collections import Counter
import json
from pathlib import Path
import re
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-kernel-loop-codegen-amd-20260922'
OUT=ROOT/'artifacts/pyannote-kernel-loop-codegen-review-20260922'


def main():
    assert not OUT.exists()
    assert pin(BASE/'closed.json')['sha256']=='bf25713d67d4a4bf72f688e0f5b92cc46bc0618afad2065dc5b7939f2d650322'
    closure=read(BASE/'closed.json');assert closure['passed']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    code=read(BASE/'listings.json');observations=[]
    for row in code['candidate']:
        if not row['tier'].startswith('Tier1'):continue
        reductions=[b for b in row['reductions'] if b['fma_instructions']>2]
        assert [b['label'] for b in reductions]==['G_M000_IG'+str(i) for i in range(21,26)]
        body=''.join(b['body'] for b in reductions)
        counts=Counter(re.findall(r'\bvfmadd\d*ps\s+(zmm\d+),',body))
        assert counts==Counter({f'zmm{i}':9 for i in range(12)})
        assert sum(b['input_broadcasts'] for b in reductions)==54
        loads=re.findall(r'\bvmovups\s+zmm(?:12|13), zmmword ptr \[(r\d+)\]',body)
        assert Counter(loads)==Counter({'r10':9,'r11':9})
        assert not re.search(r'^\s+j\w+\s',body,re.M)
        assert not any(b['vector_stack_references'] for b in reductions)
        outer=row['body'].split('G_M000_IG26:',1)[1].split('G_M000_IG28:',1)[0]
        assert re.search(r'\bcmp\s+ebx, edx',outer) and re.search(r'\bjmp\s+G_M000_IG21',outer)
        observations.append(dict(tier=row['tier'],code_bytes=row['code_bytes'],blocks=[b['label'] for b in reductions],
            ordered_accumulators=dict(counts),fmas=108,broadcasts=54,weight_loads=18,spatial_branches=0,
            vector_stack_references=0,scalar_stack_references=sum(len(b['scalar_stack_references']) for b in reductions),
            integer_multiplies=sum(b['integer_multiplies'] for b in reductions),outer_input_channel_loop_retained=True))
    assert len(observations)==2
    result=dict(passed=True,closure=pin(BASE/'closed.json'),listings=pin(BASE/'listings.json'),
        current_optimized_sizes=[r['code_bytes'] for r in code['production'] if r['tier'].startswith('Tier1')],
        candidate=observations,all_tiers_retained=True,no_performance_measurement=True,
        limitation='Static instruction counts differ in scope: current step executes nine times; candidate region contains all nine steps. Code growth and no vector spills are not speed evidence.')
    OUT.mkdir();save(OUT/'review.json',result);print(json.dumps(dict(review=pin(OUT/'review.json'),**result)))


if __name__=='__main__':main()
