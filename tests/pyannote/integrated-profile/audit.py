"""Check the two-literal adaptation, then every unchanged original audit gate."""
from common import *


def main():
    verify(read(BASE / 'source-prepared.json')['files'])
    before = (PRIOR / 'consumer/Program.cs').read_text(encoding='utf8')
    after = (BASE / 'consumer/Program.cs').read_text(encoding='utf8')
    assert before.count(OLD_CORE) == before.count(OLD_DATA) == 1
    assert after == before.replace(OLD_CORE, CORE).replace(OLD_DATA, DATA)
    for name in ['Diagnostic.cs', 'NpySupport.cs', 'SampledAudio.csproj']:
        assert pin(BASE / 'consumer' / name) == pin(PRIOR / 'consumer' / name)
    for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']:
        assert pin(BASE / 'runtime' / name) == pin(QUALIFIED / 'runtime' / name)
    original('audit_model.py')


if __name__ == '__main__':
    main()
