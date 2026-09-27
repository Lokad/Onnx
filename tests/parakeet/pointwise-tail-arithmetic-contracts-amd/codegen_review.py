"""Check the fixed eight-accumulator helpers in both observed hardware modes."""
import json
import re
from pathlib import Path
from run import BASE,ROOT,pin,read,write,prepared


def helpers(path):
    blocks=path.read_text().split('; Assembly listing for method ')[1:]
    return {b.splitlines()[0]:b for b in blocks if ':PackedColumn' in b.splitlines()[0]}


def main():
    prepared();closed=read(BASE/'closed.json');assert closed['completed'] and closed['arithmetic_contract_passed']
    for name,wanted in closed['files'].items():assert pin(BASE/name)==wanted,name
    rows=[]
    for mode in ['normal','avx512-disabled']:
        path=BASE/'capture-collected/logs'/('candidate-'+mode+'.stdout')
        before=(ROOT/'artifacts/parakeet-pointwise-tail-contracts-amd-20260927/capture-collected/logs/normal.stdout' if mode=='normal'
            else ROOT/'artifacts/parakeet-pointwise-tail-nan-diagnostic-20260927/capture-collected/logs/avx512-disabled.stdout')
        current=helpers(path);old=helpers(before);assert len(current)==len(old)==2
        for name,code in current.items():
            assert code==old[name],name
            fma=[s.strip() for s in code.splitlines() if 'vfmadd' in s]
            mul=[s.strip() for s in code.splitlines() if 'vmulps' in s]
            add=[s.strip() for s in code.splitlines() if 'vaddps' in s]
            masked='Masked' in name
            if masked:
                assert len(mul)==len(add)==8 and not fma
                assert {s.split()[1].rstrip(',') for s in add}=={'ymm'+str(i) for i in range(1,9)}
            else:
                assert len(fma)==8 and not mul and not add
                assert {s.split()[1].rstrip(',') for s in fma}=={'ymm'+str(i) for i in range(8)}
            assert not re.search(r'(?:[xyz]mmword ptr \[(?:rbp|rsp)|\[(?:rbp|rsp)[^\n]*[xyz]mm)',code)
            rows.append(dict(mode=mode,method=name,bytes=int(re.search(r'Total bytes of code (\d+)',code)[1]),
                unchanged_generated_code=True,independent_accumulators=8,fma_count=len(fma),multiply_count=len(mul),add_count=len(add),vector_stack_spills=False,disassembly=pin(path)))
    value=dict(passed=True,numerical_closure=pin(BASE/'closed.json'),helpers=rows,reviewer=pin(Path(__file__)),release_admitted=False,no_performance_measurement=True)
    write(BASE/'codegen-review.json',value)
    print(json.dumps(dict(review=pin(BASE/'codegen-review.json'),helpers=[{k:v for k,v in r.items() if k!='disassembly'} for r in rows])))


if __name__=='__main__':main()
