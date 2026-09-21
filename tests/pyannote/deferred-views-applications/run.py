"""Complete dialogue, both long meetings and recovery with original consumers."""
from common import *

original = ROOT / 'tests/pyannote/portable-applications/run.py'
source = original.read_text(encoding='utf8')
start = source.index('    for path, sha in [')
end = source.index('        if sha:', start)
source = source[:start] + '''    receipts = [(BUILD / 'closed.json', pin(BUILD / 'closed.json')['sha256']),
        (BUILD / 'focused-closed.json', '3c33d3fa6e9d2cde8877b500c7c2a68d73386040b1a1bc44b8636fea0828afd4'),
        (PRIOR / 'closed.json', '9d62d3d21a174e9be7105e4224b4b4b986b89dc6085ec60cc99b3aa2fdd7da77')]
    for name in ['shared', 'parakeet']:
        p = ROOT / ('artifacts/pyannote-deferred-views-' + name + '-20260922/closed.json')
        receipts.append((p, pin(p)['sha256']))
    assert read(BUILD / 'analysis.json')['captured_model_qualification_admitted']
    for path, sha in receipts:
''' + source[end:]
changes = [
    ("for identity in proof['identities']:", "for identity in proof.get('identities', proof.get('terminal_identities', [])):"),
    ("    assert read(TESTS / 'analysis.json')['products_unchanged']", "    assert read(BUILD / 'focused-analysis.json')['all_other_methods_and_public_declarations_unchanged']"),
    ("PRIOR / 'application-runtime'", "PRIOR / 'runtime'"),
    ('Normal source/package build with LSTM admission guard; original complete consumer and native references; no new timing comparison.',
     'One-method deferred tensor views; complete original consumer and native references; no new timing comparison.'),
    ('Complete public output/ownership/resource qualification; fixed integrated product and existing package; no new benchmark or AMD promotion.',
     'Complete public output/ownership/resource qualification of deferred views; no benchmark or AMD promotion.'),
    ('*TOOLS.iterdir(), MONITOR, INPUT,', "*TOOLS.iterdir(), *[ROOT / 'tests/pyannote/portable-applications' / n for n in ['common.py', 'run.py', 'audit.py']], MONITOR, INPUT,")]
for before, after in changes:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
namespace = dict(globals(), __name__='original_deferred_views_applications', __file__=str(Path(__file__)))
exec(compile(source, str(original), 'exec'), namespace)

if __name__ == '__main__':
    namespace['main']()
