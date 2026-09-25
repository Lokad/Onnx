"""Additive reconciliation of the JIT/CLR initial-tier naming difference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
TOOLS=ROOT/'tests/benchmarks/e5-direct-code-diagnostic-amd'
sys.path.insert(0,str(TOOLS))
import audit
from protocol import pin,read
from codegen_labels import reconcile_listings


def main():
    before=(TOOLS/'codegen.py').read_text();after=(OUT/'codegen_labels.py').read_text()
    old="""        for row,event in zip(emitted,observed,strict=True):
            p=event['payload'];assert row['runtime_tier']==p['OptimizationTier'] and row['native_bytes']==int(p['MethodSize'])
            row['event']=event"""
    new="""        for version,(row,event) in enumerate(zip(emitted,observed,strict=True)):
            p=event['payload'];actual=p['OptimizationTier']
            difference=row['runtime_tier']!=actual
            if difference:
                assert method=='RunBatchedFloatMatMul' and version==0
                assert row['tier']=='Instrumented Tier0' and actual=='QuickJitted'
                assert any('CORINFO_HELP_COUNTPROFILE32' in c for c in row['calls'])
                assert any('CORINFO_HELP_PATCHPOINT' in c for c in row['calls'])
            assert row['native_bytes']==int(p['MethodSize'])
            row['runtime_tier']=actual
            row['rendered_runtime_label_difference']=difference
            row['event']=event"""
    assert before.count(old)==1 and before.replace(old,new)==after
    prepared=read(audit.BASE/'prepared.json')
    for path in [TOOLS/'audit.py',TOOLS/'codegen.py']:
        assert pin(path)==prepared['files'][path.relative_to(ROOT).as_posix()]
    def reconcile(text,events,pid):
        result=reconcile_listings(text,events,pid)
        result['label_reconciliation']=dict(additive=True,original_parser=pin(TOOLS/'codegen.py'),
            parser=pin(OUT/'codegen_labels.py'),wrapper=pin(Path(__file__)),
            scope='Only first RunBatchedFloatMatMul listing: Instrumented Tier0 can accompany CLR QuickJitted when profile counters and OSR patchpoints are both present. All identities, order, sizes and original text remain exact.')
        return result
    audit.reconcile_listings=reconcile
    audit.main()


if __name__=='__main__':main()
