"""Keep V1 qualification exact except ten source-proven AVX512 skips."""
from pathlib import Path
from protocol import pin

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'wide-entry-first-use-root-amd'
EXP_METHODS=['ProbeHoldsContractOnWideSweep','ProbeHoldsContractOnReducedRange',
             'ProbeMatchesEstrinCore','ProbeHandlesExceptionals','ProbeTailsMatchScalar']


def expected_checks():
    value=(ORIGINAL/'checks.py').read_text()
    before="'MatMulKernelAgreementTests.SixRow512AlphaMatchesTiledAlphaBitwise']"
    assert value.count(before)==1
    after=before[:-1]+',\n'+',\n'.join("            'Exp512Tests."+name+"'" for name in EXP_METHODS)+']'
    value=value.replace(before,after)
    assert value.count('(83,3)')==1 and value.count('(3369,121)')==1
    return value.replace('(83,3)','(93,3)').replace('(3369,121)','(3359,131)')


def verify_scope():
    files={}
    for name in ['protocol.py','remote.py','audit.py','checks.py','remote_prepare.py','source_scope.py','admission.py']:
        source=ORIGINAL/name
        expected=expected_checks() if name=='checks.py' else source.read_text()
        assert (TOOLS/name).read_text()==expected,name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['run.py','prepare.py']:
        source=ORIGINAL/name
        expected=source.read_text().replace('wide-entry-first-use-root-amd-20260923','wide-entry-first-use-root-amd-v2-20260923')
        expected=expected.replace('/dev/shm/lokad-parakeet-wide-entry-first-use-root-20260923','/dev/shm/lokad-parakeet-wide-entry-first-use-root-v2-20260923')
        if name=='prepare.py':
            expected=expected.replace('from admission import product_identities','from admission import product_identities\nfrom correction import verify_incident,INCIDENT')
            expected=expected.replace("source=gates();applied=read(APPLIED/'applied.json')","source=gates();verify_incident();applied=read(APPLIED/'applied.json')")
            anchor="    copy(APPLIED/'applied.json',bundle/'evidence/root-applied.json')"
            expected=expected.replace(anchor,anchor+"\n    for name in ['closed.json','failure-analysis.json']:\n        copy(INCIDENT/name,bundle/'evidence/interrupted-root'/name)")
        assert (TOOLS/name).read_text()==expected,name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    return files


if __name__=='__main__':print(verify_scope())
