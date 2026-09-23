"""Check the only numerical-consumer changes: explicit binding and its receipt."""
from pathlib import Path


def verify_consumer():
    here = Path(__file__).resolve().parent
    old = (here.parent / 'first-use-kernels-numerics/Driver.cs').read_text()
    replacements = [
        ('    static readonly RawCall Raw = typeof(Tensor<float>).GetMethod("RunFloatMatMulKernel", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<RawCall>();',
         '    static RawCall Raw = null!;'),
        ('        string role = args[1], mode = args[2]; Role = role; int width = int.Parse(args[3]);',
         '        string role = args[1], mode = args[2]; Role = role; int width = int.Parse(args[3]);\n'
         '        Require(role is "current" or "candidate", "role");\n'
         '        string rawEntry = role == "candidate" ? "RunIsolatedShortWideKernel" : "RunFloatMatMulKernel";\n'
         '        Raw = (typeof(Tensor<float>).GetMethod(rawEntry, BindingFlags.Static | BindingFlags.NonPublic)\n'
         '            ?? throw new MissingMethodException(rawEntry)).CreateDelegate<RawCall>();'),
        ('role, mode, width, core_sha256 = core,', 'role, mode, width, core_sha256 = core, raw_entry = rawEntry,'),
    ]
    for before, after in replacements:
        assert old.count(before) == 1
        old = old.replace(before, after)
    assert old == (here / 'Driver.cs').read_text()
    return True


if __name__ == '__main__': print(verify_consumer())
