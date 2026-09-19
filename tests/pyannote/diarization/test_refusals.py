"""Deliberately corrupt optional lane evidence; preserve each rejected specimen."""
from pathlib import Path
import argparse,copy,json,os,shutil,subprocess
from audit import audit
from evidence import require

p=argparse.ArgumentParser(description=__doc__)
for name in ('reference','public','detail','runner','segmentation','encoder','projection','plda','output'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
baseline=audit(a.reference,a.public,a.detail)
require(baseline['execution_complete'],'Need complete baseline')
original=json.loads((a.reference/'manifest.json').read_text());public=json.loads(a.public.read_text());detail=json.loads((a.detail/'result.json').read_text());rows=[]

def clone_files(source,dest):
    dest.mkdir()
    for file in source.iterdir():
        if file.is_file():
            if file.suffix=='.json':shutil.copyfile(file,dest/file.name)
            else:os.link(file,dest/file.name)

mutations=[
    ('missing-case',lambda m:m['cases'].pop()),
    ('missing-file',lambda m:m['files'].pop(next(iter(m['files'])))),
    ('recipe',lambda m:m['recipes'].__setitem__('native_rules.py','0'*64)),
    ('pcm-pointer',lambda m:m['cases'][0].__setitem__('pcm','../escape.npy')),
    ('stage-pointer',lambda m:m['cases'][0]['windows'][0].__setitem__('scores',m['cases'][1]['windows'][0]['scores'])),
    ('shape',lambda m:m['files'][next(iter(m['files']))].__setitem__('shape',[1])),
    ('digest',lambda m:m['files'][next(iter(m['files']))].__setitem__('sha256','0'*64)),
]
for name,mutate in mutations:
    work=a.output/name;work.mkdir();reference=work/'reference';clone_files(a.reference,reference)
    m=copy.deepcopy(original);mutate(m);(reference/'manifest.json').write_text(json.dumps(m))
    with (work/'managed.log').open('x') as f:
        result=subprocess.run(['dotnet',str(a.runner.resolve()),str(reference.resolve()),str(a.segmentation.resolve()),str(a.encoder.resolve()),str(a.projection.resolve()),str(a.plda.resolve()),str((work/'result.json').resolve())],stdout=f,stderr=subprocess.STDOUT)
    require(result.returncode!=0 and not (work/'result.json').exists(),'Managed accepted '+name)
    try:audit(reference,a.public,a.detail)
    except (ValueError,KeyError,FileNotFoundError) as e:rows.append(dict(name=name,managed_exit=result.returncode,python=str(e)))
    else:raise ValueError('Auditor accepted '+name)

mutations=[
    ('missing-public',lambda r,d:r['reports'].pop()),
    ('duplicate-public',lambda r,d:r['reports'].append(r['reports'][0])),
    ('nan-centroid',lambda r,d:r['reports'][0]['result']['Speakers'][0]['Centroid'].__setitem__(0,float('nan'))),
    ('false-summary',lambda r,d:r.__setitem__('maximum',1.)),
    ('concurrent-output',lambda r,d:r['concurrent'][0].__setitem__('Windows',0)),
    ('missing-detail',lambda r,d:d['reports'].pop()),
    ('duplicate-detail',lambda r,d:d['reports'].append(d['reports'][0])),
    ('detail-digest',lambda r,d:d['reports'][0].__setitem__('sha256','0'*64)),
    ('detail-length',lambda r,d:d['reports'][0].__setitem__('length',1)),
    ('detail-path',lambda r,d:d['reports'][0].__setitem__('file','../escape.f32')),
    ('wrong-assembly',lambda r,d:d['assemblies']['Lokad.Onnx'].__setitem__('sha256','0'*64)),
]
for name,mutate in mutations:
    work=a.output/name;work.mkdir();target=work/'detail';clone_files(a.detail,target)
    r=copy.deepcopy(public);d=copy.deepcopy(detail);mutate(r,d)
    report=work/'public.json';report.write_text(json.dumps(r));(target/'result.json').write_text(json.dumps(d))
    try:audit(a.reference,report,target)
    except (ValueError,KeyError,FileNotFoundError) as e:rows.append(dict(name=name,python=str(e)))
    else:raise ValueError('Auditor accepted '+name)
with (a.output/'result.json').open('x') as f:json.dump(dict(passed=True,refusals=rows),f,indent=2)
print('Rejected',len(rows),'malformed evidence cases')
