"""Close with the separately documented birth-clock auditor correction."""
import audit_birth_clock
import close_phase
if __name__=='__main__':
    from pathlib import Path
    import json,sys
    base=Path(sys.argv[sys.argv.index('--artifact')+1]).resolve()
    review=json.loads((base/'birth-clock-review/receipt.json').read_text())
    root=Path(__file__).resolve().parents[3]
    for name,wanted in review['files'].items():assert audit_birth_clock.pin(root/name)==wanted,name
    close_phase.main()
