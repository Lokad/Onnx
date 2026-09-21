"""Correct archive root discovery; preserve the original failed qualification."""
import hashlib
import json
from pathlib import Path
import subprocess
from prepare import ROOT, BASE, TOOLS, SOURCE, pin, save
from run import absent

old=json.loads((BASE/'qualification.json').read_text())
assert old['complete'] and old['code']==1 and old['models']==[] and absent(old['supervisor'])
assert [(s['name'],s['code']) for s in old['steps']]==[
    ('cli-build',0),('tensors-build',0),('backend-full-tests',0),('tensors-full-tests',1)]
log=(BASE/'logs/tensors-full-tests.log').read_text()
assert 'NoOptionalParametersTests.SourceTree_HasNoOptionalParameters' in log and 'Failed:     1, Passed:   341' in log
marker=BASE/'candidate-source/Lokad.Onnx.slnx';assert not marker.exists()
raw=subprocess.check_output(['git','show',SOURCE+':Lokad.Onnx.slnx'],cwd=ROOT)
marker.write_bytes(raw)
source=TOOLS/'qualify.py';original=source.read_text()
text=original.replace("BASE/'qualification.json'","BASE/'qualification-v2.json'")
first=text.index("        for name,path in [('cli'")
last=text.index("        for name in ('Backend','Tensors'):",first)
text=text[:first]+text[last:] # Existing builds are unchanged; rerun both suites in the corrected scope.
text=text.replace("(name+'.log')","('root-corrected-'+name+'.log')")
save(BASE/'root-discovery-correction.json',dict(reason='Restore omitted source-archive solution marker; no product/consumer change',
    predecessor=pin(BASE/'qualification.json'),marker=pin(marker),source=pin(source),
    transformed_sha256=hashlib.sha256(text.encode()).hexdigest(),root_test_scan_preserved=True))
exec(compile(text,str(source), 'exec'),dict(__name__='__main__',__file__=str(source)))
