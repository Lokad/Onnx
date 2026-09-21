"""Report every observation with the original admission decision."""
from common import *

path = ROOT / 'tests/pyannote/sparse-mel-comparison/report.py'
source = path.read_text(encoding='utf8')
changes = [
    ('results-20260921.md', 'results-20260922.md'),
    ('observations-20260921.json', 'observations-20260922.json'),
    ('# Sparse mel: complete predecessor/candidate/ORT comparison', '# Deferred tensor views: complete predecessor/candidate/ORT comparison'),
    ('''Both managed roles use byte-identical Core5c0ae2aa. Data1d346664 is the accepted
dense predecessor and Datae9e4c28e the sparse candidate. All other runtime files
match. Full [application qualification](../sparse-mel-qualification/results-20260922.md)
is required before this comparison. The earlier vector-bias change is excluded.''',
     '''Predecessor Coree9c87932/Data85d166b5 is the qualified normal source build.
Candidate Core4f22824a/Data92194232 changes only RunTiledBatchFloat to defer
unused tensor wrappers. Every other Core method, all Data methods and checked
public declarations match; all other comparison runtime files are identical.
Full [application qualification](../deferred-views-applications/results-20260922.md)
is required before this comparison. Neither role includes the unselected
vector-bias experiment or the queued AVX-512 convolution dispatch.'''),
    ('artifacts/pyannote-sparse-mel-comparison-20260921', 'artifacts/pyannote-deferred-views-comparison-20260922')]
for before, after in changes:
    assert source.count(before) >= 1, before
    source = source.replace(before, after)
namespace = dict(globals(), __name__='original_deferred_views_report', __file__=str(Path(__file__)))
exec(compile(source, str(path), 'exec'), namespace)
if __name__ == '__main__':
    namespace['main']()
