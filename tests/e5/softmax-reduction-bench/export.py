"""Export all frozen phase observations after independently reproducing each verdict."""
from pathlib import Path
import argparse,json
from audit import evaluate,read,require,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();base=a.artifact
    result=dict(schema=1,bundle=read(base/'payload/bundle.json'),phases={},exporter_sha256=sha(__file__))
    phases=['control']
    if read(base/'control-audit.json')['passed']:phases.append('compare')
    for phase in phases:
        audit=read(base/(phase+'-audit.json'));collected=base/('collected-'+phase);out=collected/('result-'+phase)
        require(audit['audit_sha256']==sha(Path(__file__).with_name('audit.py')),'Auditor differs')
        require(audit['archive_sha256']==sha(base/(phase+'-results.tar.gz')) and audit['identity_sha256']==sha(out/'identity.json'),'Collected identity differs')
        identity=read(out/'identity.json');collection=collected/('collection-'+phase+'.json');require(audit['collection_sha256']==sha(collection),'Collection receipt differs')
        pins=read(collection)['files'];workers=[]
        for i in range(8):
            relative=f'result-{phase}/worker-{i}/samples.json';path=collected/relative
            require(sha(path)==pins[relative]['sha256'],'Raw sample file differs');workers.append(read(path))
        verdict=evaluate(workers,phase,audit['telemetry']);require(all(audit[k]==v for k,v in verdict.items()),'Verdict does not reproduce')
        result['phases'][phase]=dict(audit=audit,audit_file_sha256=sha(base/(phase+'-audit.json')),workers=workers,
            identities=[{k:v for k,v in row.items() if k not in ('samples','command')} for row in identity['runs']],supervisor=identity['supervisor'],supervisor_start=identity['supervisor_start'])
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,separators=(',',':'));f.write('\n')
    print(a.output,a.output.stat().st_size,sha(a.output))

if __name__=='__main__':main()
