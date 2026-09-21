"""Keep compiler identities stable and reuse the preserved negative control."""
import prepare


if __name__ == '__main__':
    prepare.BASELINE_PROOF = prepare.BASE
    prepare.BASE = prepare.ROOT/'artifacts/pyannote-lstm-panel-admission-v2-20260921'
    prepare.main()
