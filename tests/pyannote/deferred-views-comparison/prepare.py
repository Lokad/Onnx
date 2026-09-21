"""Pin the actual qualified runtimes and fresh native baseline before execution."""
from common import *


def main():
    assert len(sys.argv) == 2 and len(sys.argv[1]) == 64
    assert not BASE.exists()
    path = ROOT / 'tests/pyannote/request-comparison/prepare.py'
    source = path.read_text(encoding='utf8')
    changes = [
        ('fb7d8a4df2a7e6f51896486d1f575605f40936ebc9d3d8b373aaf2840495d10a', sys.argv[1]),
        ('c2e1d5da4746a6fb4cd4fe582663d922e074cb6bb4e03234757274cdfcf11a26', '9d62d3d21a174e9be7105e4224b4b4b986b89dc6085ec60cc99b3aa2fdd7da77'),
        ("assert read(prerequisites[0][0])['allocation_reduced']", "assert read(prerequisites[0][0])['passed']"),
        ("shutil.copytree(PREDECESSOR / 'bin', target)", "shutil.copytree(PREDECESSOR / 'runtime', target)"),
        ("            shutil.copy2(QUALIFIED / 'application-runtime/Lokad.Onnx.Data.dll', target / 'Lokad.Onnx.Data.dll')",
         "            for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']:\n                shutil.copy2(QUALIFIED / 'runtime' / name, target / name)"),
        ("assert roles[role]['Lokad.Onnx.dll']['sha256'] == '469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd'",
         "assert roles[role]['Lokad.Onnx.dll']['sha256'] == ('4f22824a7c315334982907f8846dc7e67dd0fddb1aadba4d908c89684285bbd9' if role == 'candidate' else 'e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838')"),
        ('e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb', '85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'),
        ('1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662', '92194232cd60979548cfe480db07a6f7a2e07d9f21dc8c69e133919c31ab4f2d'),
        ("p.name != 'Lokad.Onnx.Data.dll'", "p.name not in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']"),
        ("QUALIFIED / 'dialogue-output/result.json', *TOOLS.iterdir()]",
         "QUALIFIED / 'dialogue-output/result.json', *TOOLS.iterdir(), *[ROOT / 'tests/pyannote/request-comparison' / n for n in ['common.py', 'prepare.py', 'run.py', 'audit.py']]]")]
    for before, after in changes:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    namespace = dict(globals(), __name__='original_deferred_views_comparison_prepare', __file__=str(Path(__file__)))
    exec(compile(source, str(path), 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
