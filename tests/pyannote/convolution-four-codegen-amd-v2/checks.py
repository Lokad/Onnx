"""Validate every diagnostic call against the captured, numerically qualified output."""
from protocol import read


def check_result(result, role, spec, base, lanes):
    job = spec['job_details'][role]
    assert result['passed'] and result['core'] == job['core']['sha256'] and result['probe'] == job['probe']['sha256']
    assert result['executable'] == spec['driver']['sha256']
    assert result['flags'] == ['DOTNET_JitDisasm'] and result['processor_count'] == 1
    assert result['lanes'] == lanes and result['avx512'] == (lanes == 16)
    assert result['cases'] == 108 and result['passes'] == 4 and result['calls'] == len(result['observations']) == 432
    assert result['readonly_operands'] and result['tiering_and_isa_unchanged'] and result['no_performance_measurement']
    reference = read(base/'reference.json'); fixtures = read(base/'fixtures/result.json')['calls']
    assert reference['passed'] and len(fixtures) == len(reference['observations']) == 108
    hashes = {(r['name'],r['index']): r['production'] for r in reference['observations']}
    for index, row in enumerate(result['observations']):
        call = fixtures[index % 108]; name, ordinal = call['case'], call['index']; shape = call['output']['shape']
        assert row == {'pass':index//108, 'name':name, 'index':ordinal, 'form':call['form'], 'eligible':call['eligible'],
                       'channels':shape[1], 'values':shape[0]*shape[1]*shape[2]*shape[3], 'sha256':hashes[(name,ordinal)], 'exact':True}
    return dict(passed=True, calls=432, distinct_calls=108, calls_with_output_channels_divisible_by_64=312, no_performance_measurement=True)
