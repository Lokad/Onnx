"""Publish every fixed-screen clock and the unchanged admission verdict."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/ordered-wide-blocks-screen'))
from protocol import pin,read
from prepare import BASE,previous_closed
from score import ORDER,score
OUT=Path(__file__).resolve().parent
previous_closed();proof=read(BASE/'closed.json');assert proof['passed']
for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
analysis=read(BASE/'analysis.json');capture=read(BASE/'bundle/fixtures/result.json')
reports={name:read(BASE/'collected'/name/'result.json') for name in ORDER}
verdict=score(reports,capture)
for key,value in verdict.items():assert analysis[key]==value,key
outputs=['screen-clocks-20260923.csv','screen-preparation-20260923.csv','screen-observations-20260923.json','screen-20260923.md']
assert not any((OUT/name).exists() for name in outputs)
with (OUT/outputs[0]).open('w',newline='',encoding='utf8') as f,(OUT/outputs[1]).open('w',newline='',encoding='utf8') as p:
 clocks=csv.writer(f);setup=csv.writer(p)
 clocks.writerow(['process','role','index','name','node','m','reduction','columns','iteration','warmup','ticks','frequency','seconds'])
 setup.writerow(['process','index','name','node','m','reduction','columns','ticks','frequency','seconds'])
 count=0;setups=0
 for name,value in reports.items():
  frequency=value['frequency']
  for row in value['rows']:
   identity=[row[k] for k in ['index','name','node','m','reduction','columns']]
   setup.writerow([name,*identity,row['preparationTicks'],frequency,row['preparationTicks']/frequency]);setups+=1
   for clock in row['clocks']:
    clocks.writerow([name,value['role'],*identity,clock['iteration'],clock['warmup'],clock['ticks'],frequency,clock['ticks']/frequency]);count+=1
 assert count==10080 and setups==84
(OUT/outputs[2]).write_text(json.dumps(dict(closure=pin(BASE/'closed.json'),**analysis),indent=2)+'\n')
lines=['# Ordered Parakeet reduction blocks: complete-call comparison','']
lines += ['**Component admitted.** Full-model and application qualification are still required.' if analysis['admitted'] else '**Component rejected.** No application campaign or integration follows this failed gate.','']
lines += ['| Rows | Reduction | Columns | Fixture | Current ms | Candidate ms | Candidate/current | Gate |','|---:|---:|---:|---|---:|---:|---:|:---:|']
for row,entry in zip(analysis['rows'],capture['entries'],strict=True):
 kind='Derived prefix' if 'derived_prefix' in entry else 'Captured'
 lines.append(f"| {entry['m']} | {entry['k']} | {entry['n']} | {kind} | {row['current']['value']*1000:.3f} | {row['candidate']['value']*1000:.3f} | {row['ratio']['value']:.4f} | {'Pass' if row['passed'] else 'Fail'} |")
lines += ['','| Aggregate (sum of means) | Current ms | Candidate ms | Candidate/current |','|---|---:|---:|---:|']
for key,label in [('target9','Nine longer target cases'),('unchanged12','Twelve unchanged-path cases'),('all21','All 21 cases')]:
 row=analysis['scopes'][key];lines.append(f"| {label} | {row['current']['value']*1000:.3f} | {row['candidate']['value']*1000:.3f} | {row['ratio']['value']:.4f} |")
passed=sum(c['passed'] for c in analysis['controls']);gates=', '.join(g['name']+': '+('pass' if g['passed'] else 'FAIL') for g in analysis['gates'])
lines += ['',f"Repeatability: {passed}/{len(analysis['controls'])} controls pass.",f'Performance gates: {gates}.','',
 'Four fresh CPU-2 processes used current/candidate/candidate/current order,',
 'ordinary runtime flags, and 60 warmups plus 60 measurements per fixture.',
 'Every complete public MatMul call includes validation, clearing, pool rental,',
 'packing, consumption, return and tails. IO and checks remain outside timing.',
 f"All {count:,} clocks, 5,040 measurements, {setups} setup records and {analysis['resources']} resource samples are retained.",
 f"Peak owned RSS: {analysis['peak_rss']:,} bytes.",'',
 'Current Core672e5f30/Data065b7a7f; candidate Core35cbbe80/Data2515fefa.',
 'All source, numerical and generated-code prerequisites are pinned. M55 adds',
 'guarded ordered reduction blocks of 256 terms for the nine longer fixtures.',
 'Original entry, arithmetic and flags remain exact. Target-nine gain must be',
 'at least 10%; all 21 cases and all repeatability controls must pass.',
 'Root source and BENCHMARK.md remain unchanged until complete application and',
 'actual-root/package qualification. These matrix cases do not establish a',
 'complete transcription speedup or a new ORT ratio.','',
 'Evidence: artifacts/parakeet-ordered-wide-blocks-screen-amd-20260923.',
 "Closure: "+pin(BASE/'closed.json')['sha256']+'.',
 '[Every clock](screen-clocks-20260923.csv), [setup](screen-preparation-20260923.csv),',
 '[exact controls and identities](screen-observations-20260923.json).','']
(OUT/outputs[3]).write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(dict(admitted=analysis['admitted'],controls=passed,report=pin(OUT/outputs[3]),scopes=analysis['scopes'])))
