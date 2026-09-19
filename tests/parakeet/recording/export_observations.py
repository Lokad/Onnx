"""Export compact observations from closed inference outputs without running inference."""
from pathlib import Path
import argparse,hashlib,json

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();read=lambda p:json.loads(p.read_text(encoding='utf-8'));base=a.artifact
    audit=read(base/'audit.json');managed=read(base/'managed/result.json');frozen=read(base/'frozen.json');processes=read(base/'processes.json');inputs=read(base/'inputs/inputs.json')
    assert audit['complete'] and processes['complete'] and all(r['code']==0 for r in processes['runs'])
    for path,value in audit['bindings'].items():assert sha(base/path)==value
    rows=[]
    for row in managed['cases']:
        r=row['result'];windows=[]
        for w in r['windows']:
            d=w['decoding'];windows.append(dict(start_seconds=w['start_seconds'],audio_seconds=w['audio_seconds'],boundary=w['boundary'],
                text=d['text'],tokens=len(d['token_ids']),encoded_frames=d['encoded_frames'],decoder_calls=d['decoder_calls'],stop_reason=d['stop_reason'],
                decoding_sha256=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')).hexdigest()))
        rows.append(dict(name=row['name'],repeat=row['repeat'],seconds=row['seconds'],text=r['text'],duration_seconds=r['duration_seconds'],
            processed_seconds=r['processed_seconds'],stop_reason=r['stop_reason'],windows=windows))
    output=dict(schema=1,scope='Application and finite resource qualification; native reference timings include extra decoder checks and are not a baseline',
        implementation_commit=frozen['parent_commit'],audit=audit,audit_result_sha256=sha(base/'audit.json'),
        runtime=managed['runtime'],os=managed['os'],load_seconds=managed['load_seconds'],flags=managed['flags'],
        inputs=inputs,cases=rows,processes=processes,exporter_sha256=sha(__file__))
    with a.output.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2);f.write('\n')
    print(sha(a.output))

if __name__=='__main__':main()
