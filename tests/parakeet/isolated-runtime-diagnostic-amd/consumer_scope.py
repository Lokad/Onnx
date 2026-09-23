"""Preserve the screen's full fixture loop; allow only diagnostic flags/probes."""
from pathlib import Path


def verify_consumer():
    here = Path(__file__).resolve().parent
    original = (here.parent / 'isolated-short-kernels-screen/Screen.cs').read_text()
    actual = (here / 'Screen.cs').read_text()
    begin = actual.index('    static object ProbeCallers()')
    end = actual.index('    static void Main(string[] args)')
    assert begin < end and actual[begin:end].count('for (int i = 0; i < 80; i++)') == 1
    rebuilt = actual[:begin] + actual[end:]
    rebuilt = rebuilt.replace('Require(flags.Count == 1 && flags.GetValueOrDefault("DOTNET_JitDisasm") == "*MatMul* *PackPanelsB* *mm_unsafe_vectorized_intrinsics* *ShortWide*", "exact diagnostic flags");',
        'Require(flags.Count == 0, "ordinary runtime flags");')
    rebuilt = rebuilt.replace('        var callerProbes = ProbeCallers();\n', '')
    rebuilt = rebuilt.replace('protocol = "parakeet-isolated-codegen-v1",diagnosticOnly = true,codegenOnly = true,role,sequence,callerProbes,',
        'protocol = "parakeet-short-wide-complete-call-60-60-v1",role,sequence,')
    assert rebuilt == original
    assert actual.index('var callerProbes = ProbeCallers();') > actual.index('Require(index == 21,')
    return True


if __name__ == '__main__': print(verify_consumer())
