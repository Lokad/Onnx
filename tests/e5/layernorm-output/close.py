"""Close the bounded local arithmetic proof after complete independent verification."""
from pathlib import Path
import argparse,json,shutil,sys,time
import numpy as np
import audit as a

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);args=p.parse_args();base=args.artifact.resolve()
    assert not (base/'closed.json').exists()
    value=a.audit(base);assert value==a.read(base/'audit.json')
    assert 'OK' in (base/'refusal-tests-final.log').read_text()
    assert '0 Warning(s)' in (base/'build-final.log').read_text() and '0 Error(s)' in (base/'build-final.log').read_text()
    sys.path.append(str(a.ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'));import psutil
    births=[b for r in value['resources'].values() for b in r['births']]
    for item in births:
        try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
        except psutil.NoSuchProcess:pass
    generated=a.read(base/'generated/source.json')
    assert generated['source_sha256']==a.pin(a.ROOT/'src/Lokad.Onnx/TensorOps.Norm.cs')['sha256']
    assert generated['extractor_sha256']==a.pin(a.ROOT/'tests/e5/softmax-zero-blocks/generate.py')['sha256']
    assert generated['generator_sha256']==a.pin(Path(__file__).with_name('generate.py'))['sha256']
    assert generated['generated_sha256']==a.pin(base/'generated/Kernels.cs')['sha256']
    nan=a.read(base/'nan-fixture-scope.json');assert nan['passed'] is True and nan['random_signaling_nan_values']==79 and nan['random_quiet_nan_values']==81
    source=base/'closed-source';shutil.copytree(Path(__file__).parent,source,ignore=shutil.ignore_patterns('bin','obj','__pycache__','*results*.md','*observations*.json'))
    shutil.copyfile(a.ROOT/'.agent/m2-layernorm-output-20260920.md',base/'closed-plan.md')
    a.write(base/'closure-verification.json',dict(passed=True,checked_at=time.time(),audit=a.pin(base/'audit.json'),births_terminal=births,
        source_transformation=generated,nan_scope=nan,scope=value['scope']))
    files={p.relative_to(base).as_posix():a.pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    a.write(base/'closed.json',dict(schema=1,closed_at=time.time(),passed=True,scope=value['scope'],audit=a.pin(base/'audit.json'),files=files))
    print(json.dumps(dict(closed=a.pin(base/'closed.json'),files=len(files),cases=value['cases'],values=value['independent_real_values_compared'],passed=True)))

if __name__=='__main__':main()
