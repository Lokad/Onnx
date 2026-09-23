"""Preserve all 66 old groups and codegen calls; append two parallel cases."""
from pathlib import Path


def expected_consumer():
    here = Path(__file__).resolve().parent
    old = (here.parent / 'wide-projection-isolation-numerics-v2/Driver.cs').read_text()
    anchor = '    static void Codegen(JsonElement capture)'
    assert old.count(anchor) == 1
    addition = (here / 'ParallelRoutes.cs.txt').read_text()
    old = old.replace(anchor, addition + anchor)
    anchor = '        Contracts(); Require(Rows.Count == 66, "fixed numerical census");'
    assert old.count(anchor) == 1
    return old.replace(anchor, anchor + '\n        ParallelRoutes(); Require(Rows.Count == 68, "extended numerical census");')


def verify_consumer():
    here = Path(__file__).resolve().parent
    assert expected_consumer() == (here / 'Driver.cs').read_text()
    return True


if __name__ == '__main__': print(verify_consumer())
