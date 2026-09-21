"""Apply the unchanged model auditor to the corrected qualification state."""
import json
from prepare import BASE, TOOLS, pin
from run import absent

old=json.loads((BASE/'qualification.json').read_text())
correction=json.loads((BASE/'root-discovery-correction.json').read_text())
assert old['code']==1 and old['models']==[] and absent(old['supervisor'])
assert pin(BASE/'qualification.json')==correction['predecessor']
assert pin(BASE/'candidate-source/Lokad.Onnx.slnx')==correction['marker']
assert pin(TOOLS/'qualify.py')==correction['source']
source=TOOLS/'audit_qualification.py'
text=source.read_text().replace("BASE/'qualification.json'","BASE/'qualification-v2.json'")
exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
