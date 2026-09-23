"""Preserve the M54 oracle and contracts; add block boundaries, row chunks and refusals."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'wide-entry-first-use-numerics'


def expected_consumer():
    source = (PARENT / 'Driver.cs').read_text()
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    replace('    static RawCall Raw = null!;',
            '    delegate bool PackedCall(int m, int n, int k, float* a, float* packed, float* c);\n    static RawCall Raw = null!;')
    replace('(Role == "candidate" && m >= 48 && n >= 1024 && k >= 1024)',
            '(m >= 48 && n >= 1024 && k >= 1024)')
    replace('(61,4096,1024),(61,1024,4096) };',
            '(61,4096,1024),(61,1024,4096),\n            (64,4095,1024),(64,4097,1024),(66,4095,1024),(66,4097,1024),(66,1025,1024),(65,4097,1024) };')
    replace('Contracts(); Require(Rows.Count == 66, "fixed numerical census");',
            'Contracts(); Require(Rows.Count == 72, "fixed numerical census");')
    replace('ParallelRoutes(); Require(Rows.Count == 68, "extended numerical census");',
            'ParallelRoutes(); Require(Rows.Count == 76, "extended numerical census");')
    replace('foreach (int m in new[] { 64, 106 })', 'foreach (int m in new[] { 64, 106, 128, 132 })')
    replace('Role == "candidate" && m == 106 ? 4L * n * k * 4 : 0;', 'm >= 106 ? 4L * n * k * 4 : 0;')
    replace('string rawEntry = role == "candidate" ? "RunIsolatedShortWideKernel" : "RunFloatMatMulKernel";',
            'string rawEntry = "RunIsolatedShortWideKernel";')
    anchor = '    static void Codegen(JsonElement capture)'
    replace(anchor, (HERE / 'GuardRefusals.cs.txt').read_text() + anchor)
    anchor = '        if (mode == "numerics") Numerics(capture);'
    replace(anchor, '        int guardRefusals = GuardRefusals(role, mode);\n' + anchor)
    replace('role, mode, width, core_sha256 = core, raw_entry = rawEntry,',
            'role, mode, width, guardRefusals, core_sha256 = core, raw_entry = rawEntry,')
    return source


def verify_consumer():
    assert expected_consumer() == (HERE / 'Driver.cs').read_text()
    for name in ['Prototype.csproj', 'fixtures.py', 'protocol.py']:
        assert (HERE / name).read_text() == (PARENT / name).read_text(), name
    return True


if __name__ == '__main__':
    print(verify_consumer())
