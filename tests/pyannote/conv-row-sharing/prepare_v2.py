"""Preserve the initial test compile error and prepare its corrected successor."""
import json
import prepare


def main():
    failed=prepare.BASE
    builds=json.loads((failed/'builds.json').read_text())
    assert len(builds)==1 and builds[0]['name']=='backend-build' and builds[0]['code']==1
    log=(failed/'logs/backend-build.log').read_text()
    assert "error CS1061: 'Tensor<float>' does not contain a definition for 'Dims'" in log
    assert not (failed/'prepared.json').exists()
    receipt=failed/'failed-build.json'
    assert not receipt.exists()
    files={str(path.relative_to(prepare.ROOT)):prepare.pin(path) for path in failed.rglob('*')
        if path.is_file() and not {'bin','obj'}.intersection(path.relative_to(failed).parts)}
    prepare.save(receipt,dict(passed=False,stage='backend-build',product_compiled=True,tests_executed=False,
        reason='New test used internal Dims instead of public Dimensions; correct test accessor only.',files=files))
    prepare.BASE=prepare.ROOT/'artifacts/pyannote-conv-row-sharing-v2-20260921'
    prepare.main()
    prepare.save(prepare.BASE/'predecessor-failure.json',dict(path=str(receipt.relative_to(prepare.ROOT)),**prepare.pin(receipt)))


if __name__=='__main__':main()
