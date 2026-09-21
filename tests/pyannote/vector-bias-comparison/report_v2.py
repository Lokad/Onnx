"""Supply the missing console JSON import; verify already-written reports without overwriting."""
import json
from common import *

original = TOOLS / 'report.py'
namespace = dict(__name__='vector_bias_report', __file__=str(original), json=json)
exec(compile(original.read_text(encoding='utf8'), str(original), 'exec'), namespace)

if __name__ == '__main__':
    if (TOOLS / 'results-20260921.md').exists():
        analysis = read(BASE / 'analysis.json')
        closed = read(BASE / 'closed.json')
        finish = ROOT / 'artifacts/pyannote-vector-bias-comparison-finish-20260921'
        expected = dict(closure=pin(BASE / 'closed.json'), finish_closure=pin(finish / 'closed.json'), **analysis)
        assert read(TOOLS / 'observations-20260921.json') == expected
        assert pin(BASE / 'analysis.json') == closed['analysis']
        verify(closed['files'])
        text = (TOOLS / 'results-20260921.md').read_text(encoding='utf8')
        assert not analysis['attribution_valid'] and not analysis['qualifies_for_later_amd']
        assert '**fails the fixed timing repeatability controls**' in text
        for row in analysis['table']:
            for value in [*row['seconds'].values(), row['candidate_to_predecessor'], row['candidate_to_ort']]:
                assert f'{value:.6f}' in text
        for control in analysis['controls'].values():
            for value in control['fixture_max_min'].values():
                assert f'{value:.6f}' in text
        receipt = ROOT / 'artifacts/pyannote-vector-bias-report-20260921'
        receipt.mkdir()
        files = {p.relative_to(ROOT).as_posix(): pin(p) for p in [original, Path(__file__),
            TOOLS / 'results-20260921.md', TOOLS / 'observations-20260921.json', BASE / 'closed.json', BASE / 'analysis.json', finish / 'closed.json']}
        save(receipt / 'checked.json', dict(passed=True, files=files, observations_exact=True, tables_verified=True,
            original_console_print_failed=True, original_exception="NameError: name 'json' is not defined",
            explanation='Both report files were completely written before the final console print failed. This successor supplies that import and verifies the existing files without overwriting them.',
            inference_executed=False))
        print(json.dumps(dict(report_verified=True, receipt=pin(receipt / 'checked.json'))))
    else:
        namespace['main']()
