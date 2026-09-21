"""Fresh artifact after the retained zero-worker RAM preflight refusal."""
from common import *
import shutil

OUT=ROOT/'artifacts/e5-profiler-shared-v2-20260921'


def prepare():
    spec=read(BASE/'manifest.json');verify(spec);failed=read(BASE/'processes.json')
    assert failed['complete'] and failed['code']==1 and failed['runs']==[] and absent(failed['supervisor'])
    assert "'available': 9956982784" in failed['error'] and not (BASE/'outputs').exists()
    assert not OUT.exists();OUT.mkdir();shutil.copytree(BASE/'runtimes',OUT/'runtimes')
    spec['predecessor']=dict(manifest=pin(BASE/'manifest.json'),state=pin(BASE/'processes.json'),identity=failed['supervisor'],inference_calls=0)
    for path in (BASE/'manifest.json',BASE/'processes.json',Path(__file__),Path(__file__).with_name('run_wait.py'),Path(__file__).with_name('audit_successor.py')):
        spec['files'][rel(path)]=pin(path)
    for path in (OUT/'runtimes').rglob('*'):
        if path.is_file():
            original=BASE/'runtimes'/path.relative_to(OUT/'runtimes')
            assert pin(path)==pin(original);spec['files'][rel(path)]=pin(path)
    spec['preflight_wait_seconds']=3600
    write(OUT/'manifest.json',spec)
    print(json.dumps(dict(manifest=pin(OUT/'manifest.json'),predecessor=spec['predecessor'])))


if __name__=='__main__':prepare()
