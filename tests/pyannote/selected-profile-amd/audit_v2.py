"""Preserve the first parser failure, then audit the same immutable captures."""
import sys
from common import BASE,TOOLS,ROOT,pin,read,save,prepared,rel


def main():
    assert not (BASE/'closed.json').exists() and not (BASE/'audit-parser-failure.json').exists()
    spec=prepared()
    paths=[TOOLS/name for name in ['audit.py','audit_v2.py','selected_stacks.py','test_selected_stacks.py']]
    pins={rel(p):pin(p) for p in paths}
    failure=BASE/'audit-parser-failure.json'
    save(failure,dict(preserved=True,original_audit=pin(TOOLS/'audit.py'),
        error="AssertionError at stacks.py inspect: strict unnamed Thread (digits) regular expression rejected 'Thread (699331) (.NET Finalizer Thread)'",
        exports={rel(p):pin(p) for p in (BASE/'exports').rglob('*.json')},
        recovery='Accept optional nonempty exported thread label, preserve original names and every event, reject duplicate native IDs; all original accounting/coverage assertions unchanged',tools=pins))
    source=(TOOLS/'audit.py').read_text(encoding='utf8')
    old='from stacks_v2 import inspect, cross_export';assert source.count(old)==1
    source=source.replace(old,'from selected_stacks import inspect, cross_export')
    namespace=dict(__name__='selected_profile_audit_v2',__file__=str(TOOLS/'audit.py'))
    exec(compile(source,str(TOOLS/'audit.py'),'exec'),namespace)
    def checked():
        value=prepared();value['files']=dict(value['files'],**pins);value['files'][rel(failure)]=pin(failure)
        for name,wanted in pins.items():assert pin(ROOT/name)==wanted
        return value
    namespace['prepared']=checked
    namespace['main']()


if __name__=='__main__':main()
