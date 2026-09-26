"""Reuse all original identity/resource/numerical checks, with diagnostic-only output."""
from pathlib import Path
from counters import validate_reports
from prepare import ROOT, BASE


def main():
    path=ROOT/'tests/parakeet/pad-current-screen/audit.py'
    source=path.read_text(encoding='utf8')
    before='    verdict=score(reports)'
    after='''    descriptive_score=score(reports)
    memory=validate_reports(folder,reports,read)
    verdict=dict(diagnostic_only=True,admitted=False,descriptive_screen=descriptive_score,memory=memory)'''
    assert source.count(before)==1
    source=source.replace(before,after)
    namespace=dict(__name__='memory_diagnostic_audit',__file__=str(Path(__file__).resolve()),validate_reports=validate_reports)
    exec(compile(source,str(path)+' [memory diagnostic]', 'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
