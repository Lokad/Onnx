"""Admit only the fixed case/count correction; every original assertion/gate remains."""
from pathlib import Path
from protocol import pin

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
GRAPH=ROOT/'tests/parakeet/validated-composition-graphs-amd'
WARM=ROOT/'tests/benchmarks/warmed-release-amd-v2'


def expected(name):
    source=WARM/'Program.cs' if name=='Program.cs' else GRAPH/name
    text=source.read_text()
    if name=='Program.cs':
        assert text.count('3 : 780')==text.count('index < 600')==1
        text=text.replace('3 : 780','3 : 1380').replace('index < 600','index < 1200')
    elif name in ['native.py','statistics.py','test_statistics.py']:
        text=text.replace('780','1380').replace('600','1200')
    elif name=='checks.py':
        assert text.count('[(120,780),(60,600)]')==1
        text=text.replace('[(120,780),(60,600)]','[(780,1380),(600,1200)]').replace('else 780','else 1380').replace('index<600','index<1200')
    elif name=='protocol.py':
        text=text.replace("CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok','dinov3','resnet50','gpt2']","CASES=['e5-30tok']")
        text=text.replace('BUILD_JOBS=[]',"BUILD_JOBS=['sdk-version','consumer-restore','consumer-build','consumer-inventory']")
    else:assert name=='remote.py'
    return source,text


def verify():
    files={}
    for name in ['Program.cs','native.py','statistics.py','test_statistics.py','checks.py','protocol.py','remote.py']:
        source,wanted=expected(name)
        assert (TOOLS/name).read_text()==wanted,name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    return files


if __name__=='__main__':print(verify())
