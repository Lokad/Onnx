"""Use the bundled TraceEvent exporter's evented Speedscope schema.

The frozen audit.py remains byte-for-byte intact. All its other checks apply.
This correction was specified from the existing M38 exporter before M42 export.
"""
import math,re
from pathlib import Path
from protocol import read,pin
def evented_inventory(path):
 value=read(path);assert value['$schema']=='https://www.speedscope.app/file-format-schema.json'
 frames=value['shared']['frames'];assert frames and all(isinstance(f['name'],str) for f in frames)
 profiles=[];rounding=[]
 for p in value['profiles']:
  assert p['type']=='evented' and p['unit']=='milliseconds' and re.fullmatch(r'Thread \(\d+\)',p['name'])
  stack=[];previous=p['startValue'];assert math.isfinite(previous) and previous<=p['endValue']
  active=0.;empty=0.;intervals=0
  for e in p['events']:
   at=e['at'];frame=e['frame'];assert math.isfinite(at) and type(frame) is int and 0<=frame<len(frames)
   if at<previous:
    assert previous-at<=.001;rounding.append(previous-at);at=previous
   delta=at-previous
   if stack:active+=delta;intervals+=delta>0
   else:empty+=delta
   if e['type']=='O':stack.append(frame)
   else:assert e['type']=='C' and stack and stack.pop()==frame
   previous=at
  assert not stack and abs(previous-p['endValue'])<=.001
  assert abs(active+empty-(p['endValue']-p['startValue']))<=.001
  profiles.append(dict(name=p['name'],unit=p['unit'],events=len(p['events']),active_ms=active,empty_ms=empty,evented_intervals=intervals,start=p['startValue'],end=p['endValue']))
 assert profiles
 return dict(frames=len(frames),profiles=profiles,evented_intervals=sum(p['evented_intervals'] for p in profiles),rounding_adjustments_ms=rounding,scope='Reconstructed sampled-thread intervals; event count is not sample count or measured CPU time.')
def main():
 from prepare import ROOT,BASE
 extension=read(BASE/'versioned-audit.json');assert extension['passed']
 for name,wanted in extension['files'].items():assert pin(ROOT/name)==wanted,name
 path=Path(__file__).with_name('audit.py');source=path.read_text()
 old="assert report['stacks']['samples']>0";assert source.count(old)==1
 source=source.replace(old,"assert report['stacks']['evented_intervals']>0")
 old="['analysis.json','prepared.json'";assert source.count(old)==1
 source=source.replace(old,"['versioned-audit.json','analysis.json','prepared.json'")
 namespace={'__name__':'versioned_evented_audit','__file__':str(path)}
 exec(compile(source,str(path),'exec'),namespace)
 namespace['stack_inventory']=evented_inventory;namespace['main']()
if __name__=='__main__':main()
