"""Preserve the incomplete offline feed and prepare all private build packages."""
import prepare


def main():
    receipt=prepare.BASE/'failed-preparation.json'
    assert prepare.pin(receipt)['sha256']=='313b88a8a644793d50c5a1aeec32ff978f9cf8a248bcbd7649cc238707b1aea1'
    failure=prepare.read(receipt)
    assert not failure['passed'] and failure['stage']=='backend-restore' and not failure['builds_executed'] and not failure['inference_executed']
    for name,wanted in failure['files'].items():assert prepare.pin(prepare.ROOT/name)==wanted,name
    prepare.BASE=prepare.ROOT/'artifacts/pyannote-amd-candidates-v2-20260921'
    prepare.main()
    prepare.save(prepare.BASE/'predecessor-failure.json',dict(path=str(receipt.relative_to(prepare.ROOT)),**prepare.pin(receipt)))


if __name__=='__main__':main()
