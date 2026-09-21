"""Freeze the Data-only comparison after full application qualification."""
from common import *

ORIGINAL = ROOT / 'tests/pyannote/request-comparison/prepare.py'


def main():
    assert len(sys.argv) == 2 and len(sys.argv[1]) == 64
    int(sys.argv[1], 16)
    source = ORIGINAL.read_text(encoding='utf8')
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    replace('fb7d8a4df2a7e6f51896486d1f575605f40936ebc9d3d8b373aaf2840495d10a', sys.argv[1])
    replace('c2e1d5da4746a6fb4cd4fe582663d922e074cb6bb4e03234757274cdfcf11a26',
            '20fd9f38b80f2f1e2b87a375b0b2c1c99e255d77f69b674271614361e7282002')
    replace("shutil.copytree(PREDECESSOR / 'bin', target)", "shutil.copytree(PREDECESSOR / 'application-runtime', target)")
    replace('469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd',
            '5c0ae2aa7c3cce58f3ffcb190df451e053a449a3e0dbc920d7b6d0b2bc66020c')
    replace("assert roles['candidate']['Lokad.Onnx.Data.dll']['sha256'] == '1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662'",
            "assert roles['candidate']['Lokad.Onnx.Data.dll']['sha256'] == 'e9e4c28e2f7277ea226556f692d95de3d9d4a5f94eed79a9235268137dcd4775'")
    replace('e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb',
            '1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662')
    replace("QUALIFIED / 'dialogue-output/result.json', *TOOLS.iterdir()]",
            "QUALIFIED / 'dialogue-output/result.json', *TOOLS.iterdir(), *[ROOT / 'tests/pyannote/request-comparison' / n for n in ['common.py', 'prepare.py', 'run.py', 'audit.py']]]")
    replace("    assert read(prerequisites[0][0])['allocation_reduced']", "    assert read(prerequisites[0][0])['passed']  # Computation candidate: allocations are descriptive.")
    namespace = dict(globals(), __name__='original_comparison_preparation', __file__=str(Path(__file__)))
    exec(compile(source, str(ORIGINAL), 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
