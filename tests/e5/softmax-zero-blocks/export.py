"""Export audited raw samples and identities as compact, reviewable repository evidence."""
from pathlib import Path
import argparse,json
from audit import read,require,screen,sha,validate_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=a.artifact;collected=base/'collected';audit=read(base/'audit.json');identity=read(collected/'result/identity.json');workers=[]
    require(audit['audit_sha256']==sha(Path(__file__).with_name('audit.py')),'Audit source differs')
    require(audit['identity_sha256']==sha(collected/'result/identity.json'),'Worker identity differs')
    require(audit['archive_sha256']==sha(base/'results.tar.gz'),'Collected archive differs')
    require(audit['collection_sha256']==sha(collected/'collection.json'),'Terminal receipt differs')
    pins=read(collected/'collection.json')['files']
    for i,row in enumerate(identity['runs']):
        relative='result/'+row['name']+'/samples.json';path=collected/relative
        require(sha(path)==pins[relative]['sha256'],'Worker evidence differs')
        value=read(path);validate_worker(value,row['order']);workers.append(value)
    passed,table=screen(workers[:8]);require(passed==audit['prospective_timing_screen_passed'] and table==audit['results'],'Timing result does not reproduce')
    result=dict(schema=1,audit=audit,audit_file_sha256=sha(base/'audit.json'),bundle=read(collected/'bundle.json'),
        terminal=read(collected/'deployment.json'),collection_sha256=sha(collected/'collection.json'),
        worker_identities=[{k:v for k,v in r.items() if k not in ('samples','command')} for r in identity['runs']],
        workers=workers,codegen_worker_index=8,exporter_sha256=sha(__file__))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,separators=(',',':'));f.write('\n')
    print(a.output,a.output.stat().st_size,sha(a.output))


if __name__=='__main__':main()
