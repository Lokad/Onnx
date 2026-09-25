"""Keep original resource supervision and all phase/node accounting unchanged."""
import ast
from pathlib import Path

TOOLS=Path(__file__).resolve().parent
ORIGINAL=TOOLS.parent/'managed-phase-amd'


def functions(path):
    text=path.read_text(encoding='utf8')
    return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}


def verify_scope():
    before,after=functions(ORIGINAL/'remote.py'),functions(TOOLS/'remote.py')
    assert before.keys()==after.keys()
    assert [n for n in before if before[n]!=after[n]]==['build','capture']
    expected=before['capture'].replace(
        "environment=dict(env,PARAKEET_PHASE_MODE=mode,PARAKEET_PHASE_DATA_SHA=pin(runtime/'Lokad.Onnx.Data.dll')['sha256'])",
        "environment=dict(env,PARAKEET_PHASE_MODE=mode,PARAKEET_PHASE_CORE_SHA=spec['core']['sha256'],PARAKEET_PHASE_DATA_SHA=pin(runtime/'Lokad.Onnx.Data.dll')['sha256'])")
    assert after['capture']==expected
    assert functions(TOOLS/'audit.py')['attribute']==functions(ORIGINAL/'audit.py')['attribute']
    assert (TOOLS/'test_attribution.py').read_bytes()==(ORIGINAL/'test_attribution.py').read_bytes()
    return dict(passed=True,resource_supervision_unchanged=True,request_loop_unchanged=True,
        capture_changes='Explicit measured Core environment identity only',phase_node_accounting_unchanged=True)


if __name__=='__main__':print(verify_scope())
